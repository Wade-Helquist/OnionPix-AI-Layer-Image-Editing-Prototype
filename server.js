import "dotenv/config";
import express from "express";
import fs from "fs";
import multer from "multer";
import path from "path";
import sharp from "sharp";
import { fileURLToPath } from "url";

// ------------------------------------------------------
// Setup __filename and __dirname for module compatibility
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ============================================================
// HYDRA-PIX SERVER
// Pipeline Order:
//   STEP 0  - Setup (Express, Multer, API Keys, Prompts)
//   STEP 1  - Receive request, parse rectangles, clamp to image bounds
//   STEP 1.1 - Crop primary rectangle with Sharp
//   STEP 1.2 - Crop secondary rectangle with Sharp (Move Mode only)
//   STEP 2  - Label main subject via OpenAI (for preview title text)
//   STEP 2.1 - Get negative mask terms via OpenAI (scaffold below)
//   STEP 3  - Upload crop to Replicate (get URL for SAM)
//   STEP 3.1 - Run Grounded SAM (predictByVersion)
//   STEP 3.2 - Poll Replicate until job is ready (pollPrediction)
//   STEP 3.3 - Select best mask URL from SAM output (selectMaskUrl)
//   STEP 3.4 - Apply mask to crop, isolate subject (applyMaskToCrop)
//   STEP 3.4.1 - Describe isolated cutout subject (describeIsolatedCutout)
//   STEP 4  - Extend via nano-banana (scaffold below)
//   STEP 4.1 - Tighten whitespace (scaffold below)
//   STEP 5  - Download buffer from URL (downloadBuffer)
// ============================================================

// STEP 0: SETUP - initialize Express app, multer, env keys, prompts, static paths
const app = express();
app.use(express.json());
app.use("/output", express.static("output"));
const upload = multer({ dest: "uploads/" });

// STEP 0: SETUP - load and check OpenAI API key
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  throw new Error("Missing OPENAI_API_KEY");
}

// STEP 0: SETUP - load and check Replicate API key
const REPLICATE_API_KEY = process.env.REPLICATE_API_KEY;
if (!REPLICATE_API_KEY) {
  throw new Error("Missing REPLICATE_API_KEY");
}

// STEP 0: SETUP - version and prompt constants
const GROUNDED_SAM_VERSION = "ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c";
const DEFAULT_NEGATIVE_MASK_PROMPT = "";
const MAX_SEGMENTATION_ATTEMPTS = 3;

// STEP 0: SETUP - prompt texts (label, masking, extension)
const LABEL_PROMPT = `Identify the single main subject/object in this crop.
Return JSON only:
{"label":"short lower-case label (2-6 words)","title":"short readable title","description":"short plain description of the main subject"}

Selection rules (in order):
1) Prefer the subject that is most visually dominant overall in this crop
   (largest clear area + strongest focus).
2) Penalize subjects that are heavily cut off by crop edges or only partially visible.
3) Do not bias by living vs non-living; choose whichever candidate is most
   dominant and sufficiently complete in this crop.
4) If a foreground prop appears near a larger subject, choose the larger/main
   subject unless the prop is clearly more dominant and more complete.

Avoid generic labels like "thing" or "object". Be specific.
If the subject is an animal, use species if recognizable
(e.g. "clownfish" not "fish", "praying mantis" not "insect").
Label physical identity only. Do not include actions, background, effects,
fire, smoke, light, or shadows in the label.

Description rules:
- 8-20 words, plain English.
- Describe only the main subject's visible form/material/features.
- No background, scene context, effects, or other objects.`;

const LABEL_NOUN_REWRITE_PROMPT = function(label) {
  return `Rewrite this as a noun-only object/type label for segmentation masking.
Reason: this text drives mask isolation, so actions/background context harms mask quality.
Rules:
- Return plain text only (no JSON, no punctuation).
- Return only the object/type identity as a noun phrase.
- Remove actions, verbs, background/scene context, and effects.
- Keep it concise (2-6 words), lowercase.
Text: "${label}"`;
};
const NEGATIVE_PROMPT = function(label) {
  return `Main target label: "${label}"
From this crop, list other visible entities that are NOT the main target.
Include secondary people/objects/props/scene elements if visible.
Do NOT include clothing, fashion wearables, or body parts.
Keep weapons/tools/props if visible.
Return JSON array only. Example: ["person behind", "chair", "window"]`;
};

const EXTEND_PROFILE_GOLDEN = "A perfectly square 1:1 standalone asset on a pure white background. Based on the provided cutout, centrally frame a whole, seamless, fully complete version of the same subject with massive negative space around it. No part of the subject touches the canvas edges. The subject floats in unsupported levitation in a pure white void, attached to absolutely nothing. Remove all stands, bases, perches, ground surfaces, and support elements visible or implied by the cutout. Fully extend and complete all previously cut-off, hidden, missing, or obscured parts from the cutout into finished anatomical or structural form. Keep the exact material texture, color palette, and lighting style from the cutout while ignoring any local artifact contamination along its edges. No new objects, props, effects, text, or background content. Focus only on the subject's pure intact form as derived solely from the cutout.";

const EXTEND_PROMPT = function(subjectDescription = "") {
  const normalized = String(subjectDescription || "").trim();
  if (!normalized) {
    return EXTEND_PROFILE_GOLDEN;
  }
  return (
    EXTEND_PROFILE_GOLDEN +
    "\n\nMain subject description: " +
    normalized +
    ". Keep this exact subject identity while extending missing parts."
  );
};

const FULL_REMOVE_PROMPT = function(label, rect) {
  return `Remove only the ${label} near this region: x=${rect.x}, y=${rect.y}, width=${rect.width}, height=${rect.height}.
Keep every other person, object, and the background unchanged.
Do not remove anything else.
Fill the removed area naturally to match the surrounding scene and lighting.
Output the same full image dimensions with only that target removed.`;
};

const BG_CLEANUP_PROMPT =
  "Remove all subjects, figures, creatures, and objects from this image. " +
  "If there are multiple subjects or objects, remove all of them regardless of size. " +
  "If subjects are holding non-living objects such as weapons, tools, or props, remove those too. " +
  "Also remove any visually distinctive, unusual, or prominent plants that stand out clearly from the background - " +
  "such as glowing trees, giant mushrooms, or magical plants that could be considered objects. " +
  "Keep only the environment, background, scenery, and atmosphere. " +
  "Keep all manmade structures such as buildings, castles, bridges, walls, and ruins exactly identical to the original - " +
  "do not alter, add to, or remove any part of them. " +
  "Fill any areas where subjects were removed with a seamless continuation of the surrounding background. " +
  "Keep the original style, lighting, texture, and feel exactly.";

const SWAP_SUBJECT_PROMPT = function(rect, originalLabel, instructionText, footprintHint) {
  const userInstruction = String(instructionText || "").trim();
  const tweakLine = userInstruction
    ? ('Optional user tweak: "' + userInstruction + '".')
    : "Optional user tweak: none.";
  const footprintLine = footprintHint
    ? ("Original target footprint hint: " + footprintHint + ".")
    : "Original target footprint hint: use visible target scale/center from IMAGE 2.";
  return (
    "You are given three images:\n" +
    "- IMAGE 1: full original scene.\n" +
    "- IMAGE 2: extracted target subject from IMAGE 1 (to be replaced).\n" +
    "- IMAGE 3: replacement subject from library.\n" +
    "Task: swap IMAGE 2 with IMAGE 3 inside IMAGE 1.\n" +
    "Target locator region in IMAGE 1: x=" + rect.x + ", y=" + rect.y + ", width=" + rect.width + ", height=" + rect.height + ".\n" +
    "Original target label: " + String(originalLabel || "target subject") + ".\n" +
    footprintLine + "\n" +
    tweakLine + "\n" +
    "Rules:\n" +
    "- Use the locator region only to identify the original target; do not treat it as a hard crop box.\n" +
    "- Replacement can extend outside the locator if needed to naturally replace the original subject.\n" +
    "- Keep replacement centered near the original target center and at roughly the same apparent size/depth.\n" +
    "- Preserve replacement subject aspect ratio; no stretching/squashing.\n" +
    "- Keep the same scene depth layer, perspective, and placement context as the original target.\n" +
    "- Preserve existing occlusions/front-behind relationships around the target.\n" +
    "- Replace only the target subject; keep background geometry and nearby objects unchanged.\n" +
    "- Preserve IMAGE 3 subject identity, body proportions, anatomy, and defining features intact.\n" +
    "- Do not duplicate, split, merge, or invent additional body parts or structural elements.\n" +
    "- No rectangle/frame/mask borders, no sticker/matte artifacts.\n" +
    "- Keep all non-target scene elements unchanged.\n" +
    "Output one edited IMAGE 1."
  );
};

const EXTEND_DECISION_PROMPT = function(label) {
  return `You are reviewing an isolated cutout of: "${label}".
Decide if the cutout needs generative extension/fill.

Set needs_extend=true only when the subject looks incomplete, such as:
- clipped body parts/limbs at frame edges
- obvious missing sections
- interior holes/gaps that break object continuity
- heavy fragmentation into disconnected chunks

Set needs_extend=false when the subject appears complete and only minor edge cleanup is needed.

Return JSON only:
{"needs_extend":true|false,"reason":"short reason"}`;
};

const CUTOUT_DESCRIPTION_PROMPT = function(label) {
  return `You are given an isolated subject cutout for "${label}".
Describe the main subject based only on the isolated cutout image.
Rules:
- 8-20 words, plain English.
- Focus only on visible subject form/material/features.
- No background, scene, effects, or extra objects.
- Keep identity aligned to "${label}".
Return JSON only:
{"description":"short plain description of the isolated subject"}`;
};

// STEP 0: SETUP - enforce LLM response schema for labeling main subject
const LABEL_MAIN_SUBJECT_RESPONSE_FORMAT = {
  type: "json_schema",
  json_schema: {
    name: "label_response",
    strict: true,
    schema: {
      type: "object",
      additionalProperties: false,
      properties: {
        label: { type: "string" },
        title: { type: "string" },
        description: { type: "string" }
      },
      required: ["label", "title", "description"]
    }
  }
};

const EXTEND_DECISION_RESPONSE_FORMAT = {
  type: "json_schema",
  json_schema: {
    name: "extend_decision_response",
    strict: true,
    schema: {
      type: "object",
      additionalProperties: false,
      properties: {
        needs_extend: { type: "boolean" },
        reason: { type: "string" }
      },
      required: ["needs_extend", "reason"]
    }
  }
};

const CUTOUT_DESCRIPTION_RESPONSE_FORMAT = {
  type: "json_schema",
  json_schema: {
    name: "cutout_description_response",
    strict: true,
    schema: {
      type: "object",
      additionalProperties: false,
      properties: {
        description: { type: "string" }
      },
      required: ["description"]
    }
  }
};

// ============================================================
// COPY ORDER SCAFFOLD (server.js <-> ai-proof-of-concept.js)
// 1. STEP 2.1: normalizeEntityLabel
// 2. STEP 2.1: sendToOpenAI
// 3. STEP 2.1: labelNegativeMaskTerms
// 4. STEP 4:   nanoBanana
// 5. STEP 4.1: tightenWhiteSpace
// ============================================================
const COPY_ORDER_SCAFFOLD = [
  "STEP 2.1: normalizeEntityLabel",
  "STEP 2.1: sendToOpenAI",
  "STEP 2.1: labelNegativeMaskTerms",
  "STEP 4: nanoBanana",
  "STEP 4.1: tightenWhiteSpace"
];

// STEP 0: SETUP - These are referenced elsewhere
void NEGATIVE_PROMPT;
void EXTEND_PROMPT;
void FULL_REMOVE_PROMPT;
void EXTEND_DECISION_PROMPT;
void CUTOUT_DESCRIPTION_PROMPT;
void LABEL_MAIN_SUBJECT_RESPONSE_FORMAT;
void EXTEND_DECISION_RESPONSE_FORMAT;
void CUTOUT_DESCRIPTION_RESPONSE_FORMAT;
void COPY_ORDER_SCAFFOLD;

// ------------------------------------------------------
// STEP 0: SETUP - index route for root path
app.get("/", function(req, res) {
  res.sendFile(path.join(__dirname, "index.html"));
});

// STEP 1.0: LABEL PREVIEW (fast title after crop mouseup)
//   - Crop selected rect and return label/title without running full pipeline
app.post("/api/label-preview", upload.single("image"), async function(req, res) {
  try {
    const rect = JSON.parse(String(req.body.rect || "[]"));
    const offsetX = parseFloat(req.body.offsetX);
    const offsetY = parseFloat(req.body.offsetY);
    const scale = parseFloat(req.body.scale);

    if (!Array.isArray(rect) || rect.length < 4) {
      res.status(400).json({ error: "Invalid rect payload" });
      return;
    }
    if (!Number.isFinite(offsetX) || !Number.isFinite(offsetY) || !Number.isFinite(scale) || scale === 0) {
      res.status(400).json({ error: "Invalid transform values" });
      return;
    }

    const sourceBuffer = await resolveSourceImageBuffer(req);
    const metadata = await sharp(sourceBuffer).metadata();
    if (!metadata.width || !metadata.height) {
      throw new Error("Source metadata missing width/height");
    }

    const realX = Math.round((Number(rect[0]) - offsetX) / scale);
    const realY = Math.round((Number(rect[1]) - offsetY) / scale);
    const realWidth = Math.round(Number(rect[2]) / scale);
    const realHeight = Math.round(Number(rect[3]) / scale);

    if (realWidth <= 0 || realHeight <= 0) {
      res.status(400).json({ error: "Empty rectangle" });
      return;
    }

    const left = Math.max(0, realX);
    const top = Math.max(0, realY);
    const width = Math.min(realWidth, metadata.width - left);
    const height = Math.min(realHeight, metadata.height - top);
    if (width <= 0 || height <= 0) {
      res.status(400).json({ error: "Rect outside image bounds" });
      return;
    }

    const cropBuffer = await sharp(sourceBuffer)
      .extract({ left: left, top: top, width: width, height: height })
      .toBuffer();

    const result = await labelMainSubject(cropBuffer);
    res.json({
      label: result.label,
      title: result.title,
      description: result.description
    });
  } catch (error) {
    res.status(500).json({
      error: (error && error.message) ? error.message : "Label preview failed"
    });
  }
});

// STEP 1: RECEIVE REQUEST + PARSE RECTANGLES
app.post(
  "/api/rect-extract",
  upload.fields([
    { name: "image", maxCount: 1 },
    { name: "addSubjectFile", maxCount: 1 }
  ]),
  async function(req, res) {
  // I'm logging the body here for debugging
  console.log(req.body);

  // --- STEP 1A: Initialize required variables and helpers
  let primaryCropBuffer;
  let secondaryCropBuffer;
  let primaryLabel = "";
  let title = "";
  let primaryDescription = "";
  let primaryIsolatedUrl = "";
  let secondaryLabel = "";
  let secondaryTitle = "";
  let secondaryDescription = "";
  let secondaryIsolatedUrl = "";
  const steps = [];
  const requestStepId = Date.now().toString() + "-" + Math.floor(Math.random() * 1000000).toString();
  const stepOutputPath = function(filename) {
    return path.join(__dirname, "output", requestStepId + "-" + filename);
  }
  const stepOutputUrl = function(filename) {
    return "/output/" + requestStepId + "-" + filename;
  }

  // --- STEP 1B: Parse and calculate rectangle coordinates/scales
  const parseRectFromBody = function(value) {
    try {
      const parsed = JSON.parse(String(value || "[0,0,0,0]"));
      if (Array.isArray(parsed) && parsed.length >= 4) return parsed;
    } catch (error) {
      // fallback below
    }
    return [0, 0, 0, 0];
  };
  const primaryRec = parseRectFromBody(req.body.PrimaryRec);
  const secondaryRec = parseRectFromBody(req.body.SecondaryRec);
  const offsetX = parseFloat(req.body.offsetX);
  const offsetY = parseFloat(req.body.offsetY);
  const scale = parseFloat(req.body.scale);
  const mode = req.body.mode;
  const subjectAccepted = String(req.body.subjectAccepted || "false") === "true";

  // Calculate real coordinates for primary
  const realPrimaryX = Math.round((primaryRec[0] - offsetX) / scale);
  const realPrimaryY = Math.round((primaryRec[1] - offsetY) / scale);
  const realPrimaryWidth = Math.round(primaryRec[2] / scale);
  const realPrimaryHeight = Math.round(primaryRec[3] / scale);

  // Calculate for secondary if needed
  let realSecondaryX = 0;
  let realSecondaryY = 0;
  let realSecondaryWidth = 0;
  let realSecondaryHeight = 0;
  if (mode === "Move") {
    realSecondaryX = Math.round((secondaryRec[0] - offsetX) / scale);
    realSecondaryY = Math.round((secondaryRec[1] - offsetY) / scale);
    realSecondaryWidth = Math.round(secondaryRec[2] / scale);
    realSecondaryHeight = Math.round(secondaryRec[3] / scale);
  }

  // --- STEP 1C: Debug print coordinate computation
  console.log("Primary: x=", realPrimaryX, "y=", realPrimaryY, "width=", realPrimaryWidth, "height=", realPrimaryHeight);
  console.log("Secondary: x=", realSecondaryX, "y=", realSecondaryY, "width=", realSecondaryWidth, "height=", realSecondaryHeight);
  console.log("Offset X=", offsetX, "Offset Y=", offsetY, "Scale=", scale);

  // --- STEP 1D: Prepare output directory and remove old crops
  if (!fs.existsSync("output")) {
    fs.mkdirSync("output");
  }
  if (fs.existsSync("output/crop.png")) {
    fs.unlinkSync("output/crop.png");
  }
  if (fs.existsSync("output/crop-secondary.png")) {
    fs.unlinkSync("output/crop-secondary.png");
  }

  // --- STEP 1E: Detect and load source image/buffer, get metadata
  const sourceBuffer = await resolveSourceImageBuffer(req);
  const metadata = await sharp(sourceBuffer).metadata();

  if (mode === "Background") {
    const sourceUrlForBackground = await uploadToReplicate(sourceBuffer, "background-source.png");
    let cleanedBackgroundUrl;
    try {
      cleanedBackgroundUrl = await nanoBananaBackground(sourceUrlForBackground);
    } catch (error) {
      const retryAfter = Number(error && (error.retryAfter || error.waitSeconds));
      if (error && (error.code === "RATE_LIMIT" || Number.isFinite(retryAfter))) {
        res.status(429).json({
          error: "RATE_LIMITED",
          retryAfter: Number.isFinite(retryAfter) ? retryAfter : 10
        });
        return;
      }
      throw error;
    }
    const cleanedBackgroundBuffer = await downloadBuffer(cleanedBackgroundUrl);
    const cleanedStepFile = "01-background-cleaned.png";
    fs.writeFileSync(stepOutputPath(cleanedStepFile), cleanedBackgroundBuffer);
    fs.writeFileSync("output/background-final.png", cleanedBackgroundBuffer);
    steps.push({
      name: "1. Background Cleaned",
      url: stepOutputUrl(cleanedStepFile)
    });

    res.json({
      primaryCrop: "",
      primaryIsolated: stepOutputUrl(cleanedStepFile),
      secondaryCrop: "",
      secondaryIsolated: "",
      label: "",
      secondaryLabel: "",
      title: "Background Cleaned",
      description: "Generated full-scene background cleanup.",
      steps: steps,
      secondaryTitle: "",
      secondaryDescription: ""
    });
    return;
  }

  // --- STEP 1F: Clamp primary crop bounds to image bounds
  let left = Math.max(0, realPrimaryX);
  let top = Math.max(0, realPrimaryY);
  let width = Math.min(realPrimaryWidth, metadata.width - left);
  let height = Math.min(realPrimaryHeight, metadata.height - top);
  const primaryRect = { x: left, y: top, width: width, height: height };

  if (realPrimaryWidth === 0 || realPrimaryHeight === 0) {
    res.status(400).json({error: "Primary rectangle is empty"});
    return;
  }

  // STEP 1.1: CROP PRIMARY RECTANGLE - use sharp to extract main rectangle
  primaryCropBuffer = await sharp(sourceBuffer)
    .extract({left: left, top: top, width: width, height: height})
    .toBuffer();
  fs.writeFileSync("output/crop.png", primaryCropBuffer);
  const cropStepFile = "01-crop.png";
  fs.writeFileSync(stepOutputPath(cropStepFile), primaryCropBuffer);
  steps.push({
    name: "1. Crop",
    url: stepOutputUrl(cropStepFile)
  });

  // STEP 1.2: CROP SECONDARY RECTANGLE (Move Mode only) - extract if necessary
  const shouldHandleSecondary = (mode === "Move");
  const hasSecondaryRect = (realSecondaryWidth > 0 && realSecondaryHeight > 0);
  if (shouldHandleSecondary && hasSecondaryRect) {
    left = Math.max(0, realSecondaryX);
    top = Math.max(0, realSecondaryY);
    width = Math.min(realSecondaryWidth, metadata.width - left);
    height = Math.min(realSecondaryHeight, metadata.height - top);

    secondaryCropBuffer = await sharp(sourceBuffer)
      .extract({left: left, top: top, width: width, height: height})
      .toBuffer();
    fs.writeFileSync("output/crop-secondary.png", secondaryCropBuffer);
  }

  // STEP 2: LABEL PRIMARY SUBJECT - get label for main crop using OpenAI
  const resultPrimaryLabel = await labelMainSubject(primaryCropBuffer);
  primaryLabel = resultPrimaryLabel.label;
  title = resultPrimaryLabel.title;
  primaryDescription = resultPrimaryLabel.description;

  // STEP 3: UPLOAD TO REPLICATE + STEP 3.1/3.2/3.3 SAM MASK - segmentation & mask
  //   - Upload crop to Replicate
  const cropUrl = await uploadToReplicate(primaryCropBuffer, "crop.png");
  const primaryMeta = await sharp(primaryCropBuffer).metadata();
  if (!primaryMeta.width || !primaryMeta.height) {
    throw new Error("Primary crop metadata missing width/height");
  }

  let primarySegmentation;
  try {
    primarySegmentation = await segmentCropWithRetries({
      cropBuffer: primaryCropBuffer,
      cropUrl: cropUrl,
      label: primaryLabel,
      width: primaryMeta.width,
      height: primaryMeta.height
    });
  } catch (error) {
    // Rate limit handling for predictByVersion / segmentation retries
    const retryAfter = Number(error && (error.retryAfter || error.waitSeconds));
    if (error && (error.code === "RATE_LIMIT" || Number.isFinite(retryAfter))) {
      res.status(429).json({
        error: "RATE_LIMITED",
        retryAfter: Number.isFinite(retryAfter) ? retryAfter : 10
      });
      return;
    }
    throw error;
  }
  const maskBuffer = primarySegmentation.maskBuffer;
  fs.writeFileSync("output/crop-mask.png", maskBuffer);
  const maskStepFile = "02-mask.png";
  fs.writeFileSync(stepOutputPath(maskStepFile), maskBuffer);
  steps.push({
    name: "2. Mask",
    url: stepOutputUrl(maskStepFile)
  });

  // STEP 3.4: APPLY MASK TO PRIMARY CROP - cutout mask area from main crop
  const isolatedBuffer = primarySegmentation.isolatedBuffer;
  fs.writeFileSync("output/crop-isolated.png", isolatedBuffer);
  primaryIsolatedUrl = "/output/crop-isolated.png";
  const isolatedStepFile = "03-isolated.png";
  fs.writeFileSync(stepOutputPath(isolatedStepFile), isolatedBuffer);
  steps.push({
    name: "3. Isolated",
    url: stepOutputUrl(isolatedStepFile)
  });
  primaryDescription = await describeIsolatedCutout(isolatedBuffer, primaryLabel);

  // STEP 3.5 (Swap mode): swap isolated target with a library subject in one model call.
  if (mode === "Swap") {
    const swapInstruction = String(req.body.addInstruction || "").trim();
    const swapSubjectBuffer = await resolveAddSubjectBuffer(req);
    const sourceUrlForSwap = await uploadToReplicate(sourceBuffer, "swap-source.png");
    const isolatedTargetUrl = await uploadToReplicate(isolatedBuffer, "swap-target-isolated.png");
    const swapSubjectUrl = await uploadToReplicate(swapSubjectBuffer, "swap-subject.png");
    const footprintHint = await buildFootprintHintFromIsolatedBuffer(isolatedBuffer, primaryRect);
    const swapPrompt = SWAP_SUBJECT_PROMPT(
      primaryRect,
      primaryLabel,
      swapInstruction,
      footprintHint
    );
    const swappedUrl = await nanoBanana(
      [sourceUrlForSwap, isolatedTargetUrl, swapSubjectUrl],
      swapPrompt,
      "swap-subject"
    );
    const swappedRawBuffer = await downloadBuffer(swappedUrl);
    const swappedFinalBuffer = await applyLocalizedSwapFromGenerated({
      sourceBuffer: sourceBuffer,
      generatedBuffer: swappedRawBuffer,
      targetIsolatedBuffer: isolatedBuffer,
      rect: primaryRect,
      fullWidth: metadata.width,
      fullHeight: metadata.height
    });

    const swapRawStepFile = "04-swapped-raw.png";
    fs.writeFileSync(stepOutputPath(swapRawStepFile), swappedRawBuffer);
    steps.push({
      name: "4. Swapped (Raw)",
      url: stepOutputUrl(swapRawStepFile)
    });

    const swapFinalStepFile = "05-swapped.png";
    fs.writeFileSync(stepOutputPath(swapFinalStepFile), swappedFinalBuffer);
    steps.push({
      name: "5. Swapped",
      url: stepOutputUrl(swapFinalStepFile)
    });

    res.json({
      primaryCrop: "/output/crop.png",
      primaryIsolated: stepOutputUrl(swapFinalStepFile),
      secondaryCrop: "",
      secondaryIsolated: "",
      label: "",
      secondaryLabel: "",
      title: "Swapped Subject",
      description: swapInstruction
        ? "Swapped selected subject using library replacement and user tweak."
        : "Swapped selected subject using library replacement.",
      steps: steps,
      secondaryTitle: "",
      secondaryDescription: "",
      addPlacement: {
        x: primaryRect.x,
        y: primaryRect.y,
        width: primaryRect.width,
        height: primaryRect.height
      }
    });
    return;
  }

  const needsSubjectAcceptance = (mode === "Extract" || mode === "Remove");
  if (needsSubjectAcceptance && !subjectAccepted) {
    res.json({
      primaryCrop: "/output/crop.png",
      primaryIsolated: primaryIsolatedUrl,
      secondaryCrop: hasSecondaryRect ? "/output/crop-secondary.png" : "",
      secondaryIsolated: secondaryIsolatedUrl,
      label: primaryLabel,
      secondaryLabel: secondaryLabel,
      title: title,
      description: primaryDescription,
      steps: steps,
      secondaryTitle: secondaryTitle,
      secondaryDescription: secondaryDescription,
      awaitingSubjectAcceptance: true
    });
    return;
  }

  // STEP 4 + STEP 4.1: FILL/EXTEND PIPELINE (Extract mode)
  //   - Always run extend, then tighten
  if (mode === "Extract") {
    const isolatedUrl = await uploadToReplicate(isolatedBuffer, "crop-isolated.png");
    let extendedImageUrl;
    const extendSubjectDescription = String(primaryDescription || title || primaryLabel || "").trim();

    try {
      extendedImageUrl = await nanoBanana(
        isolatedUrl,
        EXTEND_PROMPT(extendSubjectDescription),
        primaryLabel
      );
    } catch (error) {
      const retryAfter = Number(error && (error.retryAfter || error.waitSeconds));
      if (error && (error.code === "RATE_LIMIT" || Number.isFinite(retryAfter))) {
        res.status(429).json({
          error: "RATE_LIMITED",
          retryAfter: Number.isFinite(retryAfter) ? retryAfter : 10
        });
        return;
      }
      throw error;
    }

    const rawExtendedBuffer = await downloadBuffer(extendedImageUrl);
    const stage4Buffer = await constrainExtendedToOriginal(
      isolatedBuffer,
      rawExtendedBuffer,
      primaryMeta.width,
      primaryMeta.height
    );
    fs.writeFileSync("output/crop-extended.png", stage4Buffer);
    const extendedStepFile = "04-extended.png";
    fs.writeFileSync(stepOutputPath(extendedStepFile), stage4Buffer);
    steps.push({
      name: "4. Extend",
      url: stepOutputUrl(extendedStepFile)
    });

    const tightenedBuffer = await tightenWhiteSpace(stage4Buffer, 8);
    fs.writeFileSync("output/crop-final-tight.png", tightenedBuffer);
    primaryIsolatedUrl = "/output/crop-final-tight.png";
    const tightenedStepFile = "05-tightened.png";
    fs.writeFileSync(stepOutputPath(tightenedStepFile), tightenedBuffer);
    steps.push({
      name: "5. Tighten",
      url: stepOutputUrl(tightenedStepFile)
    });
    // Refresh subject description from the final post-extend cutout result.
    primaryDescription = await describeIsolatedCutout(tightenedBuffer, primaryLabel);
  }

  // STEP 4 (Remove mode): PREP FULL-CANVAS INVERTED MASK AND INPAINT
  //   - Do not run subject extend pipeline in Remove mode
  if (mode === "Remove") {
    const removeHoleAlpha = await buildFullCanvasHoleAlpha(
      maskBuffer,
      primaryRect,
      metadata.width,
      metadata.height
    );

    const removeMaskBuffer = await buildFullCanvasSubjectMask(
      maskBuffer,
      primaryRect,
      metadata.width,
      metadata.height
    );
    fs.writeFileSync("output/remove-mask-full.png", removeMaskBuffer);
    const removeMaskStepFile = "04-remove-mask.png";
    fs.writeFileSync(stepOutputPath(removeMaskStepFile), removeMaskBuffer);
    steps.push({
      name: "4. Remove Mask",
      url: stepOutputUrl(removeMaskStepFile)
    });

    const removePrepBuffer = await buildRemovePrepCanvas(
      sourceBuffer,
      maskBuffer,
      primaryRect,
      metadata.width,
      metadata.height
    );
    fs.writeFileSync("output/remove-prep.png", removePrepBuffer);
    const removePrepStepFile = "05-remove-prep.png";
    fs.writeFileSync(stepOutputPath(removePrepStepFile), removePrepBuffer);
    steps.push({
      name: "5. Remove Prep",
      url: stepOutputUrl(removePrepStepFile)
    });

    const removePrepUrl = await uploadToReplicate(removePrepBuffer, "remove-prep.png");
    let removedImageUrl;
    try {
      removedImageUrl = await nanoBanana(
        removePrepUrl,
        FULL_REMOVE_PROMPT(primaryLabel, primaryRect),
        primaryLabel + " remove"
      );
    } catch (error) {
      const retryAfter = Number(error && (error.retryAfter || error.waitSeconds));
      if (error && (error.code === "RATE_LIMIT" || Number.isFinite(retryAfter))) {
        res.status(429).json({
          error: "RATE_LIMITED",
          retryAfter: Number.isFinite(retryAfter) ? retryAfter : 10
        });
        return;
      }
      throw error;
    }

    const removedRawBuffer = await downloadBuffer(removedImageUrl);
    const removedCanvasBuffer = await mergeInpaintIntoSource(
      sourceBuffer,
      removedRawBuffer,
      removeHoleAlpha,
      metadata.width,
      metadata.height
    );
    fs.writeFileSync("output/remove-final.png", removedCanvasBuffer);
    const removedStepFile = "06-removed.png";
    fs.writeFileSync(stepOutputPath(removedStepFile), removedCanvasBuffer);
    steps.push({
      name: "6. Removed",
      url: stepOutputUrl(removedStepFile)
    });
    primaryIsolatedUrl = stepOutputUrl(removedStepFile);
  }

  // STEP 2: LABEL SECONDARY SUBJECT (Move Mode only) - get label if needed
  if (secondaryCropBuffer) {
    const resultSecondaryLabel = await labelMainSubject(secondaryCropBuffer);
    secondaryLabel = resultSecondaryLabel.label;
    secondaryTitle = resultSecondaryLabel.title;
    secondaryDescription = resultSecondaryLabel.description;

    const secondaryCropUrl = await uploadToReplicate(
      secondaryCropBuffer,
      "crop-secondary.png"
    );

    const secondaryMeta = await sharp(secondaryCropBuffer).metadata();
    if (!secondaryMeta.width || !secondaryMeta.height) {
      throw new Error("Secondary crop metadata missing width/height");
    }

    let secondarySegmentation;
    try {
      secondarySegmentation = await segmentCropWithRetries({
        cropBuffer: secondaryCropBuffer,
        cropUrl: secondaryCropUrl,
        label: secondaryLabel,
        width: secondaryMeta.width,
        height: secondaryMeta.height
      });
    } catch (error) {
      const retryAfter = Number(error && (error.retryAfter || error.waitSeconds));
      if (error && (error.code === "RATE_LIMIT" || Number.isFinite(retryAfter))) {
        res.status(429).json({
          error: "RATE_LIMITED",
          retryAfter: Number.isFinite(retryAfter) ? retryAfter : 10
        });
        return;
      }
      throw error;
    }

    const secondaryMaskBuffer = secondarySegmentation.maskBuffer;
    fs.writeFileSync("output/crop-secondary-mask.png", secondaryMaskBuffer);
    const secondaryIsolatedBuffer = secondarySegmentation.isolatedBuffer;
    fs.writeFileSync("output/crop-secondary-isolated.png", secondaryIsolatedBuffer);
    secondaryIsolatedUrl = "/output/crop-secondary-isolated.png";
  }

  // --- STEP 1G: Send all image step outputs and labels for frontend processing
  res.json({
    primaryCrop: "/output/crop.png",
    primaryIsolated: primaryIsolatedUrl,
    secondaryCrop: hasSecondaryRect ? "/output/crop-secondary.png" : "",
    secondaryIsolated: secondaryIsolatedUrl,
    label: primaryLabel,
    secondaryLabel: secondaryLabel,
    title: title,
    description: primaryDescription,
    steps: steps,
    secondaryTitle: secondaryTitle,
    secondaryDescription: secondaryDescription
  });
});

//  Step 1.3 Resolve Source Image buffer
//   - Return buffer from uploaded file or image URL
async function resolveSourceImageBuffer(req) {
  if (req.file && req.file.path) {
    return fs.readFileSync(req.file.path);
  }
  if (
    req.files &&
    req.files.image &&
    req.files.image[0] &&
    req.files.image[0].path
  ) {
    return fs.readFileSync(req.files.image[0].path);
  }
  if (req.body && req.body.sourceImageUrl) {
    return await downloadBuffer(req.body.sourceImageUrl);
  }
  throw new Error("Source image buffer not found");
}

// STEP 1.3: Resolve swap subject buffer
//   - Return buffer from uploaded addSubjectFile or addSubjectUrl
async function resolveAddSubjectBuffer(req) {
  if (
    req.files &&
    req.files.addSubjectFile &&
    req.files.addSubjectFile[0] &&
    req.files.addSubjectFile[0].path
  ) {
    return fs.readFileSync(req.files.addSubjectFile[0].path);
  }
  const addSubjectUrl = String((req.body && req.body.addSubjectUrl) || "").trim();
  if (addSubjectUrl) {
    return await downloadBuffer(addSubjectUrl);
  }
  throw new Error("Swap mode missing addSubjectFile/addSubjectUrl");
}

// STEP 2: LABEL MAIN SUBJECT (OpenAI)
//   - Send cropped buffer for GPT-4o vision labeling (OpenAI)
async function labelMainSubject(cropBuffer) {
  const b64 = cropBuffer.toString("base64");
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      max_tokens: 120,
      response_format: LABEL_MAIN_SUBJECT_RESPONSE_FORMAT,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: LABEL_PROMPT },
            {
              type: "image_url",
              image_url: { url: "data:image/png;base64," + b64 }
            }
          ]
        }
      ]
    })
  });
  const data = await res.json();
  // not sure if content is always here so I should check
  let content;
  if (data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) {
    content = data.choices[0].message.content;
  } else {
    throw new Error("OpenAI label response missing content (status " + res.status + ")");
  }
  const parsed = JSON.parse(content);
  // Step 2.1: ensure returned label is a noun phrase
  const nounLabel = await rewriteLabelToNounsOnly(parsed.label);
  const description = String(parsed.description || "")
    .replace(/[\r\n]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return {
    label: nounLabel,
    title: toTitleCase(nounLabel),
    description: description || toTitleCase(nounLabel)
  };
}

// STEP 3.4.1: DESCRIBE ISOLATED CUTOUT (OpenAI)
//   - Generate subject description from post-mask cutout only
async function describeIsolatedCutout(isolatedBuffer, label) {
  const b64 = isolatedBuffer.toString("base64");
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      max_tokens: 100,
      response_format: CUTOUT_DESCRIPTION_RESPONSE_FORMAT,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: CUTOUT_DESCRIPTION_PROMPT(label) },
            {
              type: "image_url",
              image_url: { url: "data:image/png;base64," + b64 }
            }
          ]
        }
      ]
    })
  });

  const data = await res.json();
  let content;
  if (data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) {
    content = data.choices[0].message.content;
  } else {
    throw new Error("OpenAI cutout description response missing content (status " + res.status + ")");
  }

  const parsed = JSON.parse(content);
  const description = String(parsed.description || "")
    .replace(/[\r\n]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!description) {
    throw new Error("OpenAI cutout description returned empty description");
  }

  return description;
}

// STEP 2.1: Rewrite subject label to nouns only (noun phrase, for mask prompt)
async function rewriteLabelToNounsOnly(labelText) {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      max_tokens: 32,
      messages: [
        {
          role: "user",
          content: LABEL_NOUN_REWRITE_PROMPT(String(labelText || ""))
        }
      ]
    })
  });
  const data = await res.json();
  let content;
  if (data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) {
    content = data.choices[0].message.content;
  } else {
    throw new Error("OpenAI noun rewrite response missing content (status " + res.status + ")");
  }

  // I think this just cleans the label to get rid of whitespace and punctuation
  let cleaned = String(content)
    .replace(/[\r\n]+/g, " ")
    .replace(/^["'`]+|["'`]+$/g, "")
    .replace(/[.!,;:]+$/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();

  if (!cleaned) {
    throw new Error("OpenAI noun rewrite returned empty label");
  }

  return cleaned;
}

// STEP 3.4.2: ASK AI IF EXTEND/FILL IS NEEDED
//   - Vision decision on missing limbs/holes/fragmentation before Step 4
async function assessExtendNeedWithAI(isolatedBuffer, label) {
  const b64 = isolatedBuffer.toString("base64");
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "gpt-4o-mini",
      max_tokens: 80,
      response_format: EXTEND_DECISION_RESPONSE_FORMAT,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: EXTEND_DECISION_PROMPT(label) },
            {
              type: "image_url",
              image_url: { url: "data:image/png;base64," + b64 }
            }
          ]
        }
      ]
    })
  });

  const data = await res.json();
  let content;
  if (data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) {
    content = data.choices[0].message.content;
  } else {
    throw new Error("OpenAI extend decision response missing content (status " + res.status + ")");
  }

  const parsed = JSON.parse(content);
  return {
    needsExtend: Boolean(parsed.needs_extend),
    reason: String(parsed.reason || "")
  };
}

// Utility: Convert phrase to title case for display
function toTitleCase(text) {
  // this will make the first letter in each word uppercase
  return String(text || "")
    .split(" ")
    .filter(function(w) { return Boolean(w); })
    .map(function(w) { return w[0].toUpperCase() + w.slice(1); })
    .join(" ");
}

// getMimeFromPath - returns MIME type string based on file extension
function getMimeFromPath(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".jpg" || ext === ".jpeg") {
    return "image/jpeg";
  }
  if (ext === ".webp") {
    return "image/webp";
  }
  if (ext === ".bmp") {
    return "image/bmp";
  }
  return "image/png";
}

// STEP 3: UPLOAD TO REPLICATE
//   - Upload file buffer as proper blob for Replicate's API, return a URL
async function uploadToReplicate(buffer, filename) {
  // I have to use Blob and FormData because Replicate expects real files I guess
  const blob = new Blob([buffer], { type: getMimeFromPath(filename) });
  const form = new FormData();
  form.append("content", blob, path.basename(filename));
  const res = await fetch("https://api.replicate.com/v1/files", {
    method: "POST",
    headers: { Authorization: `Bearer ${REPLICATE_API_KEY}` },
    body: form
  });
  const data = await res.json();
  console.log("Status: " + data.status);
  console.log("Error: " + data.error);
  console.log("URLs: " + (data.urls ? data.urls.get : ""));
  return data.urls.get;
}

// STEP 3.1: RUN GROUNDED SAM
//   - Post prediction request to Replicate's API for segmentation/mask
async function predictByVersion(version, input, label) {
  const run = async function() {
    return fetch("https://api.replicate.com/v1/predictions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${REPLICATE_API_KEY}`,
        "Content-Type": "application/json",
        Prefer: "wait"
      },
      body: JSON.stringify({ version: version, input: input })
    });
  }

  let res = await run();
  let data = await res.json();
  // Handle rate limiting if present
  if (
    res.status === 429 ||
    String(data.detail || "").toLowerCase().includes("throttled")
  ) {
    const waitSeconds = Number(data.retry_after || 10);
    console.log("   rate limited for " + label + ", waiting " + waitSeconds + "s");
    const error = new Error("Rate limited for " + label);
    error.code = "RATE_LIMIT";
    error.retryAfter = waitSeconds;
    error.waitSeconds = waitSeconds;
    throw error;
  }

  if (data.error) {
    throw new Error("Prediction error (" + label + "): " + data.error);
  }
  // If not complete, poll for result
  if (data.id && ["succeeded", "failed", "canceled"].indexOf(data.status) === -1) {
    data = await pollPrediction(data.id);
  }
  console.log("Status: " + data.status);
  console.log("Error: " + data.error);
  console.log("Output: " + data.output);
  if (data.status !== "succeeded") {
    throw new Error("Prediction did not succeed (" + label + ")");
  }
  return data.output;
}

// STEP 3.2: POLL REPLICATE UNTIL READY
//   - Poll Replicate until the operation is finished, succeed/fail/cancel
async function pollPrediction(id) {
  while (true) {
    const res = await fetch("https://api.replicate.com/v1/predictions/" + id, {
      headers: { Authorization: `Bearer ${REPLICATE_API_KEY}` }
    });
    const data = await res.json();
    if (data.status === "succeeded") {
      return data;
    }
    if (data.status === "failed" || data.status === "canceled") {
      throw new Error("Prediction " + id + " failed: " + (data.error || data.status));
    }
    await new Promise(function(r) { setTimeout(r, 1500); });
  }
}

// STEP 3.3: SELECT BEST MASK URL
//   - Try to pick the appropriate mask url from Replicate SAM outputs
function listOutputUrls(outputUrls) {
  let urls = [];
  if (Array.isArray(outputUrls)) {
    urls = outputUrls.map(function(v) { return String(v || ""); });
  } else {
    urls = [String(outputUrls || "")];
  }
  return urls
    .map(function(v) { return v.trim(); })
    .filter(function(v) { return Boolean(v); });
}

function selectMaskUrlCandidates(outputUrls) {
  const urls = listOutputUrls(outputUrls);
  const chosen = [];
  const seen = new Set();

  const add = function(url) {
    if (!url || seen.has(url)) return;
    chosen.push(url);
    seen.add(url);
  };

  urls
    .filter(function(u) { return /(^|\/)mask\.(png|jpg|jpeg)/i.test(u); })
    .forEach(add);
  urls
    .filter(function(u) { return (/mask/i.test(u) && !/inverted/i.test(u)); })
    .forEach(add);
  urls
    .filter(function(u) { return /mask/i.test(u); })
    .forEach(add);
  urls.forEach(add);

  return chosen;
}

function selectMaskUrl(outputUrls) {
  const candidates = selectMaskUrlCandidates(outputUrls);
  if (!candidates[0]) throw new Error("grounded_sam returned no output URLs");
  return candidates[0];
}

async function needsSegmentationRetry(isolatedBuffer) {
  const raw = await sharp(isolatedBuffer)
    .ensureAlpha()
    .raw()
    .toBuffer();

  let visiblePixelCount = 0;
  let nonWhiteVisiblePixelCount = 0;

  for (let i = 0; i < raw.length; i += 4) {
    const r = raw[i];
    const g = raw[i + 1];
    const b = raw[i + 2];
    const a = raw[i + 3];

    if (a <= 8) continue;
    visiblePixelCount += 1;
    if (!(r === 255 && g === 255 && b === 255)) {
      nonWhiteVisiblePixelCount += 1;
    }
  }

  if (visiblePixelCount === 0) {
    return { retry: true, reason: "transparent result" };
  }
  if (nonWhiteVisiblePixelCount === 0) {
    return { retry: true, reason: "pure white result" };
  }
  return { retry: false, reason: "" };
}

// STEP 3.4.1: RETRY SEGMENTATION IF OUTPUT IS EMPTY
//   - Retry SAM/mask selection if isolated result is transparent or pure white
async function segmentCropWithRetries(options) {
  const cropBuffer = options.cropBuffer;
  const cropUrl = options.cropUrl;
  const label = options.label;
  const width = options.width;
  const height = options.height;

  let lastResult = null;

  for (let attempt = 1; attempt <= MAX_SEGMENTATION_ATTEMPTS; attempt += 1) {
    const output = await predictByVersion(
      GROUNDED_SAM_VERSION,
      {
        image: cropUrl,
        mask_prompt: label,
        negative_mask_prompt: DEFAULT_NEGATIVE_MASK_PROMPT
      },
      label
    );

    const maskCandidates = selectMaskUrlCandidates(output);
    if (!maskCandidates.length) {
      throw new Error("No mask candidates returned for " + label);
    }

    for (const maskUrl of maskCandidates) {
      const maskBuffer = await downloadBuffer(maskUrl);
      const isolatedBuffer = await applyMaskToCrop(cropBuffer, maskBuffer, width, height);
      const quality = await needsSegmentationRetry(isolatedBuffer);

      lastResult = {
        maskUrl: maskUrl,
        maskBuffer: maskBuffer,
        isolatedBuffer: isolatedBuffer
      };

      if (!quality.retry) {
        return lastResult;
      }

      console.log(
        "Retry needed for " + label +
        " (attempt " + attempt + "/" + MAX_SEGMENTATION_ATTEMPTS + "): " +
        quality.reason
      );
    }
  }

  if (lastResult) {
    console.log("Segmentation retries exhausted for " + label + ", using last result");
    return lastResult;
  }

  throw new Error("Segmentation produced no usable result for " + label);
}

// STEP 3.4: APPLY MASK TO CROP
//   - Use mask buffer as alpha mask for the crop buffer to isolate main subject
async function applyMaskToCrop(cropBuffer, maskBuffer, width, height) {
  // I want to create an RGBA mask based on gray threshold
  const alpha = await sharp(maskBuffer)
    .resize(width, height, { fit: "fill" })
    .grayscale()
    .threshold(24)
    .raw()
    .toBuffer();

  const rgbaMask = await sharp({
    create: {
      width: width,
      height: height,
      channels: 3,
      background: { r: 255, g: 255, b: 255 }
    }
  })
    .joinChannel(alpha, { raw: { width: width, height: height, channels: 1 }})
    .png()
    .toBuffer();

  return sharp(cropBuffer)
    .resize(width, height, { fit: "fill" })
    .composite([{ input: rgbaMask, blend: "dest-in" }])
    .png()
    .toBuffer();
}

// STEP 3.4.4: BUILD FULL-CANVAS SUBJECT MASK (Remove mode visualization)
//   - Place crop-space subject mask onto original full canvas coordinates
async function buildFullCanvasSubjectMask(maskBuffer, rect, fullWidth, fullHeight) {
  const subjectAlphaCrop = await sharp(maskBuffer)
    .resize(rect.width, rect.height, { fit: "fill" })
    .grayscale()
    .threshold(24)
    .raw()
    .toBuffer();

  const rgb = Buffer.alloc(fullWidth * fullHeight * 3, 255);
  for (let y = 0; y < rect.height; y += 1) {
    const dstY = rect.y + y;
    if (dstY < 0 || dstY >= fullHeight) continue;
    for (let x = 0; x < rect.width; x += 1) {
      const dstX = rect.x + x;
      if (dstX < 0 || dstX >= fullWidth) continue;
      const srcIdx = (y * rect.width) + x;
      if (subjectAlphaCrop[srcIdx] <= 0) continue;
      const dstPx = (dstY * fullWidth) + dstX;
      const dst = dstPx * 3;
      rgb[dst] = 0;
      rgb[dst + 1] = 0;
      rgb[dst + 2] = 0;
    }
  }

  return sharp(rgb, { raw: { width: fullWidth, height: fullHeight, channels: 3 } })
    .png()
    .toBuffer();
}

// STEP 3.4.4: BUILD FULL-CANVAS REMOVE HOLE ALPHA
//   - Create subject-hole alpha in full-canvas coordinates (subject=255, else 0)
async function buildFullCanvasHoleAlpha(maskBuffer, rect, fullWidth, fullHeight) {
  const subjectAlphaCrop = await sharp(maskBuffer)
    .resize(rect.width, rect.height, { fit: "fill" })
    .grayscale()
    .threshold(24)
    .raw()
    .toBuffer();

  const holeAlpha = Buffer.alloc(fullWidth * fullHeight, 0);
  for (let y = 0; y < rect.height; y += 1) {
    const dstY = rect.y + y;
    if (dstY < 0 || dstY >= fullHeight) continue;
    for (let x = 0; x < rect.width; x += 1) {
      const dstX = rect.x + x;
      if (dstX < 0 || dstX >= fullWidth) continue;
      const srcIdx = (y * rect.width) + x;
      const dstIdx = (dstY * fullWidth) + dstX;
      holeAlpha[dstIdx] = subjectAlphaCrop[srcIdx];
    }
  }
  return holeAlpha;
}

// STEP 3.4.4: BUILD REMOVE PREP ON MAIN CANVAS
//   - Expand inverted crop mask to full canvas, then white-overlay removed subject area
async function buildRemovePrepCanvas(sourceBuffer, maskBuffer, rect, fullWidth, fullHeight) {
  const holeAlpha = await buildFullCanvasHoleAlpha(
    maskBuffer,
    rect,
    fullWidth,
    fullHeight
  );

  const whiteHoleOverlay = await sharp({
    create: {
      width: fullWidth,
      height: fullHeight,
      channels: 3,
      background: { r: 255, g: 255, b: 255 }
    }
  })
    .joinChannel(holeAlpha, { raw: { width: fullWidth, height: fullHeight, channels: 1 } })
    .png()
    .toBuffer();

  // Overlay on the full original canvas so all remove prep work happens on main image coordinates.
  return sharp(sourceBuffer)
    .resize(fullWidth, fullHeight, { fit: "fill" })
    .ensureAlpha()
    .composite([{ input: whiteHoleOverlay, blend: "over" }])
    .png()
    .toBuffer();
}

// STEP 3.4.5: MERGE INPAINTED REMOVE AREA ONLY
//   - Keep source pixels outside remove-hole to prevent cumulative quality loss
async function mergeInpaintIntoSource(sourceBuffer, inpaintBuffer, holeAlpha, fullWidth, fullHeight) {
  const sourceRgba = await sharp(sourceBuffer)
    .resize(fullWidth, fullHeight, { fit: "fill" })
    .ensureAlpha()
    .raw()
    .toBuffer();

  const inpaintRgba = await sharp(inpaintBuffer)
    .resize(fullWidth, fullHeight, { fit: "fill" })
    .ensureAlpha()
    .raw()
    .toBuffer();

  const out = Buffer.alloc(sourceRgba.length);
  const alphaThreshold = 8;
  for (let i = 0; i < holeAlpha.length; i += 1) {
    const outIdx = i * 4;
    if (holeAlpha[i] > alphaThreshold) {
      out[outIdx] = inpaintRgba[outIdx];
      out[outIdx + 1] = inpaintRgba[outIdx + 1];
      out[outIdx + 2] = inpaintRgba[outIdx + 2];
      out[outIdx + 3] = 255;
    } else {
      out[outIdx] = sourceRgba[outIdx];
      out[outIdx + 1] = sourceRgba[outIdx + 1];
      out[outIdx + 2] = sourceRgba[outIdx + 2];
      out[outIdx + 3] = sourceRgba[outIdx + 3];
    }
  }

  return sharp(out, { raw: { width: fullWidth, height: fullHeight, channels: 4 } })
    .png()
    .toBuffer();
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}


async function applyRectOnlyFromGenerated(options) {
  const sourceBuffer = options.sourceBuffer;
  const generatedBuffer = options.generatedBuffer;
  const rect = options.rect;
  const fullWidth = Number(options.fullWidth);
  const fullHeight = Number(options.fullHeight);

  const x = clampNumber(Math.round(rect.x), 0, Math.max(0, fullWidth - 1));
  const y = clampNumber(Math.round(rect.y), 0, Math.max(0, fullHeight - 1));
  const w = clampNumber(Math.round(rect.width), 1, fullWidth - x);
  const h = clampNumber(Math.round(rect.height), 1, fullHeight - y);

  const sourceSized = await sharp(sourceBuffer)
    .resize(fullWidth, fullHeight, { fit: "fill" })
    .ensureAlpha()
    .png()
    .toBuffer();

  const generatedSized = await sharp(generatedBuffer)
    .resize(fullWidth, fullHeight, { fit: "fill" })
    .ensureAlpha()
    .png()
    .toBuffer();

  const generatedRect = await sharp(generatedSized)
    .extract({ left: x, top: y, width: w, height: h })
    .png()
    .toBuffer();

  return sharp(sourceSized)
    .composite([{ input: generatedRect, left: x, top: y, blend: "over" }])
    .png()
    .toBuffer();
}

async function buildFootprintHintFromIsolatedBuffer(isolatedBuffer, rect) {
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  const alpha = await sharp(isolatedBuffer)
    .ensureAlpha()
    .extractChannel("alpha")
    .resize(w, h, { fit: "fill" })
    .raw()
    .toBuffer();

  let minX = w;
  let minY = h;
  let maxX = -1;
  let maxY = -1;
  let count = 0;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const a = alpha[(y * w) + x];
      if (a <= 20) continue;
      count += 1;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }

  if (count === 0 || maxX < minX || maxY < minY) {
    return "";
  }

  const bw = Math.max(1, maxX - minX + 1);
  const bh = Math.max(1, maxY - minY + 1);
  const cx = rect.x + minX + Math.round(bw / 2);
  const cy = rect.y + minY + Math.round(bh / 2);
  return "center≈(" + cx + "," + cy + "), size≈" + bw + "x" + bh;
}

async function applyLocalizedSwapFromGenerated(options) {
  const sourceBuffer = options.sourceBuffer;
  const generatedBuffer = options.generatedBuffer;
  const rect = options.rect;
  const fullWidth = Number(options.fullWidth);
  const fullHeight = Number(options.fullHeight);

  const sourceRaw = await sharp(sourceBuffer)
    .resize(fullWidth, fullHeight, { fit: "fill" })
    .ensureAlpha()
    .raw()
    .toBuffer();

  const generatedRaw = await sharp(generatedBuffer)
    .resize(fullWidth, fullHeight, { fit: "fill" })
    .ensureAlpha()
    .raw()
    .toBuffer();

  const pixelCount = fullWidth * fullHeight;
  const diffMask = new Uint8Array(pixelCount);
  const diffThreshold = 24;
  for (let i = 0; i < pixelCount; i += 1) {
    const px = i * 4;
    const d =
      Math.abs(generatedRaw[px] - sourceRaw[px]) +
      Math.abs(generatedRaw[px + 1] - sourceRaw[px + 1]) +
      Math.abs(generatedRaw[px + 2] - sourceRaw[px + 2]);
    diffMask[i] = d >= diffThreshold ? 1 : 0;
  }

  // Locator expansion allows natural subject growth beyond the drawn box while
  // blocking unrelated scene-wide drift.
  const rx = clampNumber(Math.round(rect.x), 0, Math.max(0, fullWidth - 1));
  const ry = clampNumber(Math.round(rect.y), 0, Math.max(0, fullHeight - 1));
  const rw = clampNumber(Math.round(rect.width), 1, fullWidth - rx);
  const rh = clampNumber(Math.round(rect.height), 1, fullHeight - ry);
  const padX = Math.max(24, Math.round(rw * 0.8));
  const padY = Math.max(24, Math.round(rh * 0.8));
  const sx = clampNumber(rx - padX, 0, fullWidth - 1);
  const sy = clampNumber(ry - padY, 0, fullHeight - 1);
  const ex = clampNumber(rx + rw + padX, 0, fullWidth);
  const ey = clampNumber(ry + rh + padY, 0, fullHeight);

  const intersectsExpandedLocator = function(minX, minY, maxX, maxY) {
    return !(maxX < sx || maxY < sy || minX >= ex || minY >= ey);
  };

  const keepMask = new Uint8Array(pixelCount);
  const visited = new Uint8Array(pixelCount);
  const minComponentArea = Math.max(20, Math.round((rw * rh) * 0.0010));

  for (let y = 0; y < fullHeight; y += 1) {
    for (let x = 0; x < fullWidth; x += 1) {
      const start = (y * fullWidth) + x;
      if (!diffMask[start] || visited[start]) continue;

      const stack = [start];
      const pixels = [];
      visited[start] = 1;
      let minX = x;
      let minY = y;
      let maxX = x;
      let maxY = y;

      while (stack.length > 0) {
        const idx = stack.pop();
        pixels.push(idx);
        const cx = idx % fullWidth;
        const cy = Math.floor(idx / fullWidth);

        if (cx < minX) minX = cx;
        if (cy < minY) minY = cy;
        if (cx > maxX) maxX = cx;
        if (cy > maxY) maxY = cy;

        if (cx > 0) {
          const n = idx - 1;
          if (diffMask[n] && !visited[n]) { visited[n] = 1; stack.push(n); }
        }
        if (cx < fullWidth - 1) {
          const n = idx + 1;
          if (diffMask[n] && !visited[n]) { visited[n] = 1; stack.push(n); }
        }
        if (cy > 0) {
          const n = idx - fullWidth;
          if (diffMask[n] && !visited[n]) { visited[n] = 1; stack.push(n); }
        }
        if (cy < fullHeight - 1) {
          const n = idx + fullWidth;
          if (diffMask[n] && !visited[n]) { visited[n] = 1; stack.push(n); }
        }
      }

      if (
        pixels.length >= minComponentArea &&
        intersectsExpandedLocator(minX, minY, maxX, maxY)
      ) {
        for (const idx of pixels) keepMask[idx] = 255;
      }
    }
  }

  let keptCount = 0;
  for (let i = 0; i < keepMask.length; i += 1) {
    if (keepMask[i] > 0) keptCount += 1;
  }

  if (keptCount < Math.max(48, Math.round((rw * rh) * 0.02))) {
    // If localized filtering found too little, return generated result to avoid
    // "flash then disappear" behavior.
    return sharp(generatedBuffer)
      .resize(fullWidth, fullHeight, { fit: "fill" })
      .png()
      .toBuffer();
  }

  const keepMaskBuffer = await sharp(Buffer.from(keepMask), {
    raw: { width: fullWidth, height: fullHeight, channels: 1 }
  })
    .blur(1.8)
    .raw()
    .toBuffer();

  const out = Buffer.from(sourceRaw);
  for (let i = 0; i < pixelCount; i += 1) {
    const px = i * 4;
    const a = keepMaskBuffer[i] / 255;
    if (a <= 0.001) continue;
    out[px] = Math.round((sourceRaw[px] * (1 - a)) + (generatedRaw[px] * a));
    out[px + 1] = Math.round((sourceRaw[px + 1] * (1 - a)) + (generatedRaw[px + 1] * a));
    out[px + 2] = Math.round((sourceRaw[px + 2] * (1 - a)) + (generatedRaw[px + 2] * a));
    out[px + 3] = 255;
  }

  return sharp(out, {
    raw: { width: fullWidth, height: fullHeight, channels: 4 }
  })
    .png()
    .toBuffer();
}

async function applyTargetMaskLimitedFromGenerated(options) {
  const sourceBuffer = options.sourceBuffer;
  const generatedBuffer = options.generatedBuffer;
  const targetIsolatedBuffer = options.targetIsolatedBuffer;
  const rect = options.rect;
  const fullWidth = Number(options.fullWidth);
  const fullHeight = Number(options.fullHeight);

  const x = clampNumber(Math.round(rect.x), 0, Math.max(0, fullWidth - 1));
  const y = clampNumber(Math.round(rect.y), 0, Math.max(0, fullHeight - 1));
  const w = clampNumber(Math.round(rect.width), 1, fullWidth - x);
  const h = clampNumber(Math.round(rect.height), 1, fullHeight - y);

  const sourceSized = await sharp(sourceBuffer)
    .resize(fullWidth, fullHeight, { fit: "fill" })
    .ensureAlpha()
    .png()
    .toBuffer();

  const sourceRectRaw = await sharp(sourceSized)
    .extract({ left: x, top: y, width: w, height: h })
    .raw()
    .toBuffer();

  const generatedRectRaw = await sharp(generatedBuffer)
    .resize(fullWidth, fullHeight, { fit: "fill" })
    .ensureAlpha()
    .extract({ left: x, top: y, width: w, height: h })
    .raw()
    .toBuffer();

  const targetMaskRaw = await sharp(targetIsolatedBuffer)
    .ensureAlpha()
    .extractChannel("alpha")
    .resize(w, h, { fit: "fill" })
    .raw()
    .toBuffer();

  const diffMask = new Uint8Array(w * h);
  const diffThreshold = 36;
  for (let i = 0; i < w * h; i += 1) {
    const px = i * 4;
    const d =
      Math.abs(generatedRectRaw[px] - sourceRectRaw[px]) +
      Math.abs(generatedRectRaw[px + 1] - sourceRectRaw[px + 1]) +
      Math.abs(generatedRectRaw[px + 2] - sourceRectRaw[px + 2]);
    diffMask[i] = d >= diffThreshold ? 1 : 0;
  }

  const visited = new Uint8Array(w * h);
  let best = null;
  for (let yy = 0; yy < h; yy += 1) {
    for (let xx = 0; xx < w; xx += 1) {
      const start = (yy * w) + xx;
      if (!diffMask[start] || visited[start]) continue;

      const stack = [start];
      visited[start] = 1;
      let minX = xx;
      let minY = yy;
      let maxX = xx;
      let maxY = yy;
      let area = 0;
      let overlap = 0;

      while (stack.length > 0) {
        const idx = stack.pop();
        const cx = idx % w;
        const cy = Math.floor(idx / w);
        area += 1;
        if (targetMaskRaw[idx] > 20) overlap += 1;
        if (cx < minX) minX = cx;
        if (cy < minY) minY = cy;
        if (cx > maxX) maxX = cx;
        if (cy > maxY) maxY = cy;

        if (cx > 0) {
          const n = idx - 1;
          if (diffMask[n] && !visited[n]) { visited[n] = 1; stack.push(n); }
        }
        if (cx < w - 1) {
          const n = idx + 1;
          if (diffMask[n] && !visited[n]) { visited[n] = 1; stack.push(n); }
        }
        if (cy > 0) {
          const n = idx - w;
          if (diffMask[n] && !visited[n]) { visited[n] = 1; stack.push(n); }
        }
        if (cy < h - 1) {
          const n = idx + w;
          if (diffMask[n] && !visited[n]) { visited[n] = 1; stack.push(n); }
        }
      }

      const component = {
        minX: minX,
        minY: minY,
        maxX: maxX,
        maxY: maxY,
        area: area,
        overlap: overlap,
        score: (overlap * 4) + area
      };
      if (!best || component.score > best.score) best = component;
    }
  }

  let tMinX = w;
  let tMinY = h;
  let tMaxX = -1;
  let tMaxY = -1;
  for (let yy = 0; yy < h; yy += 1) {
    for (let xx = 0; xx < w; xx += 1) {
      const idx = (yy * w) + xx;
      if (targetMaskRaw[idx] <= 20) continue;
      if (xx < tMinX) tMinX = xx;
      if (yy < tMinY) tMinY = yy;
      if (xx > tMaxX) tMaxX = xx;
      if (yy > tMaxY) tMaxY = yy;
    }
  }

  if (!best || tMaxX < tMinX || tMaxY < tMinY) {
    return sourceSized;
  }

  const genBoxW = Math.max(1, best.maxX - best.minX + 1);
  const genBoxH = Math.max(1, best.maxY - best.minY + 1);
  const targetBoxW = Math.max(1, tMaxX - tMinX + 1);
  const targetBoxH = Math.max(1, tMaxY - tMinY + 1);

  const genPatchRgb = await sharp(generatedRectRaw, {
    raw: { width: w, height: h, channels: 4 }
  })
    .extract({ left: best.minX, top: best.minY, width: genBoxW, height: genBoxH })
    .removeAlpha()
    .raw()
    .toBuffer();

  const genPatchMask = Buffer.alloc(genBoxW * genBoxH);
  for (let py = 0; py < genBoxH; py += 1) {
    for (let px = 0; px < genBoxW; px += 1) {
      const srcIdx = ((best.minY + py) * w) + (best.minX + px);
      const dstIdx = (py * genBoxW) + px;
      genPatchMask[dstIdx] = diffMask[srcIdx] ? 255 : 0;
    }
  }

  const genPatchRgba = await sharp(genPatchRgb, {
    raw: { width: genBoxW, height: genBoxH, channels: 3 }
  })
    .joinChannel(genPatchMask, { raw: { width: genBoxW, height: genBoxH, channels: 1 } })
    .png()
    .toBuffer();

  const scaledPatch = await sharp(genPatchRgba)
    .resize(targetBoxW, targetBoxH, { fit: "inside" })
    .png()
    .toBuffer();

  const scaledMeta = await sharp(scaledPatch).metadata();
  const sw = Math.max(1, Number(scaledMeta.width || 1));
  const sh = Math.max(1, Number(scaledMeta.height || 1));
  const placeX = x + tMinX + Math.round((targetBoxW - sw) / 2);
  const placeY = y + tMinY + Math.round((targetBoxH - sh) / 2);

  return sharp(sourceSized)
    .composite([{ input: scaledPatch, left: placeX, top: placeY, blend: "over" }])
    .png()
    .toBuffer();
}

// STEP 3.4.3: HEURISTIC QUALITY ANALYSIS (fallback)
//   - Detect likely incompleteness from border touches, holes, and fragmentation
async function analyzeIsolationQuality(buffer, width, height) {
  const alphaRaw = await sharp(buffer)
    .ensureAlpha()
    .extractChannel("alpha")
    .raw()
    .toBuffer();

  const alphaThreshold = 16;
  const borderPad = Math.max(1, Math.round(Math.min(width, height) * 0.02));
  let visiblePixelCount = 0;
  let borderVisiblePixelCount = 0;
  let leftBorderVisible = 0;
  let rightBorderVisible = 0;
  let topBorderVisible = 0;
  let bottomBorderVisible = 0;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (y * width) + x;
      const alpha = alphaRaw[idx];
      if (alpha <= alphaThreshold) continue;

      visiblePixelCount += 1;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      if (x < borderPad) leftBorderVisible += 1;
      if (x >= width - borderPad) rightBorderVisible += 1;
      if (y < borderPad) topBorderVisible += 1;
      if (y >= height - borderPad) bottomBorderVisible += 1;
      if (
        x < borderPad ||
        x >= width - borderPad ||
        y < borderPad ||
        y >= height - borderPad
      ) {
        borderVisiblePixelCount += 1;
      }
    }
  }

  if (visiblePixelCount === 0) {
    return {
      needsExtend: true,
      reason: "Empty/transparent cutout",
      fillRatio: 0,
      borderTouchRatio: 1,
      componentCount: 0,
      holeRatio: 1
    };
  }

  const bboxWidth = Math.max(1, maxX - minX + 1);
  const bboxHeight = Math.max(1, maxY - minY + 1);
  const fillRatio = visiblePixelCount / (bboxWidth * bboxHeight);
  const borderTouchRatio = borderVisiblePixelCount / visiblePixelCount;

  // Build a small alpha map for component/hole checks.
  const sampleMax = 96;
  const sampleWidth = Math.max(8, Math.min(sampleMax, width));
  const sampleHeight = Math.max(8, Math.min(sampleMax, height));
  const sampleAlpha = await sharp(alphaRaw, { raw: { width: width, height: height, channels: 1 } })
    .resize(sampleWidth, sampleHeight, { fit: "fill" })
    .raw()
    .toBuffer();

  const sampleSize = sampleWidth * sampleHeight;
  const foreground = new Uint8Array(sampleSize);
  let foregroundCount = 0;
  for (let i = 0; i < sampleSize; i += 1) {
    if (sampleAlpha[i] > alphaThreshold) {
      foreground[i] = 1;
      foregroundCount += 1;
    }
  }

  const minComponentSize = Math.max(4, Math.floor(foregroundCount * 0.01));
  const componentVisited = new Uint8Array(sampleSize);
  let componentCount = 0;
  let largestComponentSize = 0;

  for (let i = 0; i < sampleSize; i += 1) {
    if (!foreground[i] || componentVisited[i]) continue;

    const stack = [i];
    componentVisited[i] = 1;
    let size = 0;

    while (stack.length > 0) {
      const idx = stack.pop();
      size += 1;
      const x = idx % sampleWidth;
      const y = Math.floor(idx / sampleWidth);

      if (x > 0) {
        const leftIdx = idx - 1;
        if (foreground[leftIdx] && !componentVisited[leftIdx]) {
          componentVisited[leftIdx] = 1;
          stack.push(leftIdx);
        }
      }
      if (x < sampleWidth - 1) {
        const rightIdx = idx + 1;
        if (foreground[rightIdx] && !componentVisited[rightIdx]) {
          componentVisited[rightIdx] = 1;
          stack.push(rightIdx);
        }
      }
      if (y > 0) {
        const upIdx = idx - sampleWidth;
        if (foreground[upIdx] && !componentVisited[upIdx]) {
          componentVisited[upIdx] = 1;
          stack.push(upIdx);
        }
      }
      if (y < sampleHeight - 1) {
        const downIdx = idx + sampleWidth;
        if (foreground[downIdx] && !componentVisited[downIdx]) {
          componentVisited[downIdx] = 1;
          stack.push(downIdx);
        }
      }
    }

    if (size >= minComponentSize) {
      componentCount += 1;
      if (size > largestComponentSize) {
        largestComponentSize = size;
      }
    }
  }

  const largestComponentRatio = foregroundCount > 0 ? (largestComponentSize / foregroundCount) : 0;

  // Flood-fill exterior background, then remaining background pixels are interior holes.
  const backgroundVisited = new Uint8Array(sampleSize);
  const queue = [];
  const pushBg = function(idx) {
    if (idx < 0 || idx >= sampleSize) return;
    if (foreground[idx] || backgroundVisited[idx]) return;
    backgroundVisited[idx] = 1;
    queue.push(idx);
  };

  for (let x = 0; x < sampleWidth; x += 1) {
    pushBg(x);
    pushBg(((sampleHeight - 1) * sampleWidth) + x);
  }
  for (let y = 0; y < sampleHeight; y += 1) {
    pushBg(y * sampleWidth);
    pushBg((y * sampleWidth) + (sampleWidth - 1));
  }

  while (queue.length > 0) {
    const idx = queue.pop();
    const x = idx % sampleWidth;
    const y = Math.floor(idx / sampleWidth);

    if (x > 0) pushBg(idx - 1);
    if (x < sampleWidth - 1) pushBg(idx + 1);
    if (y > 0) pushBg(idx - sampleWidth);
    if (y < sampleHeight - 1) pushBg(idx + sampleWidth);
  }

  let holePixelCount = 0;
  for (let i = 0; i < sampleSize; i += 1) {
    if (!foreground[i] && !backgroundVisited[i]) {
      holePixelCount += 1;
    }
  }
  const holeRatio = foregroundCount > 0 ? (holePixelCount / foregroundCount) : 0;

  const sideTouchLimit = Math.max(4, Math.round(visiblePixelCount * 0.01));
  const clippedSides = [leftBorderVisible, rightBorderVisible, topBorderVisible, bottomBorderVisible]
    .filter(function(count) { return count >= sideTouchLimit; })
    .length;

  const borderBad = borderTouchRatio > 0.06 || clippedSides >= 1;
  const holeBad = holeRatio > 0.08;
  const fragmentedBad = componentCount >= 2 && largestComponentRatio < 0.95;
  const sparseBad = fillRatio < 0.2;
  const needsExtend = borderBad || holeBad || fragmentedBad || sparseBad;

  const reasons = [];
  if (borderBad) reasons.push("Touches border/cut off");
  if (holeBad) reasons.push("Interior holes");
  if (fragmentedBad) reasons.push("Fragmented mask");
  if (sparseBad) reasons.push("Low subject fill");
  if (reasons.length === 0) reasons.push("Cutout appears complete");

  return {
    needsExtend: needsExtend,
    reason: reasons.join(", "),
    fillRatio: fillRatio,
    borderTouchRatio: borderTouchRatio,
    componentCount: componentCount,
    holeRatio: holeRatio,
    borderBad: borderBad,
    holeBad: holeBad,
    fragmentedBad: fragmentedBad,
    sparseBad: sparseBad
  };
}

// STEP 4.0: EDGE CLEANUP (no-extend path)
//   - Smooth alpha edge noise while preserving transparent background
async function cleanIsolatedEdges(buffer, width, height) {
  const alphaRaw = await sharp(buffer)
    .ensureAlpha()
    .extractChannel("alpha")
    .median(1)
    .blur(0.35)
    .raw()
    .toBuffer();

  const rgbRaw = await sharp(buffer)
    .ensureAlpha()
    .removeAlpha()
    .raw()
    .toBuffer();

  return sharp(rgbRaw, { raw: { width: width, height: height, channels: 3 } })
    .joinChannel(alphaRaw, { raw: { width: width, height: height, channels: 1 } })
    .png()
    .toBuffer();
}

// STEP 4.0.1: CONSTRAIN EXTEND OUTPUT TO ORIGINAL SUBJECT
//   - Keep original subject pixels locked and allow new pixels only near silhouette growth
async function constrainExtendedToOriginal(
  originalBuffer,
  extendedBuffer,
  width,
  height
) {
  const originalRgba = await sharp(originalBuffer)
    .resize(width, height, { fit: "fill" })
    .ensureAlpha()
    .raw()
    .toBuffer();

  const extendedRgba = await sharp(extendedBuffer)
    .resize(width, height, { fit: "fill" })
    .ensureAlpha()
    .raw()
    .toBuffer();
  const alphaThreshold = 16;
  const pixelCount = width * height;
  const componentIds = new Int32Array(pixelCount);
  componentIds.fill(-1);

  // Build connected components on extended alpha mask.
  const components = [];
  let nextId = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (y * width) + x;
      const a = extendedRgba[(idx * 4) + 3];
      if (a <= alphaThreshold || componentIds[idx] !== -1) continue;

      const id = nextId++;
      const stack = [idx];
      componentIds[idx] = id;
      let size = 0;
      let overlap = 0;

      while (stack.length > 0) {
        const cur = stack.pop();
        size += 1;
        const curAlphaOriginal = originalRgba[(cur * 4) + 3];
        if (curAlphaOriginal > alphaThreshold) overlap += 1;

        const cx = cur % width;
        const cy = Math.floor(cur / width);

        if (cx > 0) {
          const left = cur - 1;
          if (componentIds[left] === -1 && extendedRgba[(left * 4) + 3] > alphaThreshold) {
            componentIds[left] = id;
            stack.push(left);
          }
        }
        if (cx < width - 1) {
          const right = cur + 1;
          if (componentIds[right] === -1 && extendedRgba[(right * 4) + 3] > alphaThreshold) {
            componentIds[right] = id;
            stack.push(right);
          }
        }
        if (cy > 0) {
          const up = cur - width;
          if (componentIds[up] === -1 && extendedRgba[(up * 4) + 3] > alphaThreshold) {
            componentIds[up] = id;
            stack.push(up);
          }
        }
        if (cy < height - 1) {
          const down = cur + width;
          if (componentIds[down] === -1 && extendedRgba[(down * 4) + 3] > alphaThreshold) {
            componentIds[down] = id;
            stack.push(down);
          }
        }
      }

      components.push({ id: id, size: size, overlap: overlap });
    }
  }

  // Keep extended components that materially overlap the original subject.
  const keepIds = new Set();
  for (const c of components) {
    const minOverlap = Math.max(4, Math.round(c.size * 0.01));
    if (c.overlap >= minOverlap) {
      keepIds.add(c.id);
    }
  }

  const out = Buffer.alloc(originalRgba.length);
  for (let i = 0; i < originalRgba.length; i += 4) {
    const px = i / 4;
    const originalAlpha = originalRgba[i + 3];
    const extendedAlpha = extendedRgba[i + 3];
    const sameSubjectComponent = keepIds.has(componentIds[px]);

    if (sameSubjectComponent && extendedAlpha > alphaThreshold) {
      out[i] = extendedRgba[i];
      out[i + 1] = extendedRgba[i + 1];
      out[i + 2] = extendedRgba[i + 2];
      out[i + 3] = extendedAlpha;
    } else if (originalAlpha > alphaThreshold) {
      // Preserve original subject pixels if extend dropped them.
      out[i] = originalRgba[i];
      out[i + 1] = originalRgba[i + 1];
      out[i + 2] = originalRgba[i + 2];
      out[i + 3] = originalAlpha;
    } else {
      out[i] = 255;
      out[i + 1] = 255;
      out[i + 2] = 255;
      out[i + 3] = 0;
    }
  }

  return sharp(out, { raw: { width: width, height: height, channels: 4 } })
    .png()
    .toBuffer();
}

// STEP 4: SEEDREAM-4
//   - Send image + prompt to Seedream 4 model for fill/extend operations
async function nanoBanana(imageInput, prompt, label) {
  const isThrottled = function(response, body) {
    return (
      response.status === 429 ||
      String((body && body.detail) || "").toLowerCase().includes("throttled")
    );
  };

  const run = async function() {
    return fetch("https://api.replicate.com/v1/models/bytedance/seedream-4/predictions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${REPLICATE_API_KEY}`,
        "Content-Type": "application/json",
        Prefer: "wait"
      },
      body: JSON.stringify({
        input: {
          image_input: Array.isArray(imageInput) ? imageInput : [imageInput],
          prompt: prompt,
          size: "1K",
          aspect_ratio: "match_input_image",
          sequential_image_generation: "disabled",
          max_images: 1,
          enhance_prompt: false
        }
      })
    });
  };

  let res = await run();
  let data = await res.json();

  if (isThrottled(res, data)) {
    const waitSeconds = Number(data.retry_after || 10);
    console.log("   rate limited for " + label + ", waiting " + waitSeconds + "s");
    await new Promise(function(resolve) { setTimeout(resolve, waitSeconds * 1000); });
    res = await run();
    data = await res.json();
  }
  if (isThrottled(res, data)) {
    const waitSeconds = Number(data.retry_after || 10);
    const error = new Error("Rate limited for " + label);
    error.code = "RATE_LIMIT";
    error.retryAfter = waitSeconds;
    error.waitSeconds = waitSeconds;
    throw error;
  }

  if (data.error) {
    throw new Error("Prediction error (" + label + "): " + data.error);
  }
  if (data.id && ["succeeded", "failed", "canceled"].indexOf(data.status) === -1) {
    data = await pollPrediction(data.id);
  }
  if (data.status !== "succeeded") {
    throw new Error("Prediction did not succeed (" + label + ")");
  }

  let output = data.output;
  if (Array.isArray(output)) {
    output = output[0];
  }
  if (!output) {
    throw new Error("No output URL from seedream-4 (" + label + ")");
  }
  return output;
}

// STEP 4B: NANO-BANANA BACKGROUND CLEANUP
//   - Dedicated full-scene background cleanup model call
async function nanoBananaBackground(sourceImageUrl) {
  const isThrottled = function(response, body) {
    return (
      response.status === 429 ||
      String((body && body.detail) || "").toLowerCase().includes("throttled")
    );
  };

  const run = async function() {
    return fetch("https://api.replicate.com/v1/models/google/nano-banana/predictions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${REPLICATE_API_KEY}`,
        "Content-Type": "application/json",
        Prefer: "wait"
      },
      body: JSON.stringify({
        input: {
          image_input: [sourceImageUrl],
          prompt: BG_CLEANUP_PROMPT,
          output_format: "png"
        }
      })
    });
  };

  let res = await run();
  let data = await res.json();

  if (isThrottled(res, data)) {
    const waitSeconds = Number(data.retry_after || 10);
    await new Promise(function(resolve) { setTimeout(resolve, waitSeconds * 1000); });
    res = await run();
    data = await res.json();
  }

  if (isThrottled(res, data)) {
    const waitSeconds = Number(data.retry_after || 10);
    const error = new Error("Rate limited for background cleanup");
    error.code = "RATE_LIMIT";
    error.retryAfter = waitSeconds;
    error.waitSeconds = waitSeconds;
    throw error;
  }

  if (data.error) {
    throw new Error("Prediction error (background cleanup): " + data.error);
  }
  if (data.id && ["succeeded", "failed", "canceled"].indexOf(data.status) === -1) {
    data = await pollPrediction(data.id);
  }
  if (data.status !== "succeeded") {
    throw new Error("Prediction did not succeed (background cleanup)");
  }

  let output = data.output;
  if (Array.isArray(output)) {
    output = output[0];
  }
  if (!output) {
    throw new Error("No output URL from nano-banana (background cleanup)");
  }
  return output;
}

// STEP 4.1: TIGHTEN WHITESPACE
//   - Trim extra white border and add consistent white padding
async function tightenWhiteSpace(buffer, pad = 8) {
  const trimmed = await sharp(buffer)
    .trim({
      background: { r: 255, g: 255, b: 255 },
      threshold: 10
    })
    .png()
    .toBuffer();

  return sharp(trimmed)
    .extend({
      top: pad,
      bottom: pad,
      left: pad,
      right: pad,
      background: { r: 255, g: 255, b: 255, alpha: 1 }
    })
    .png()
    .toBuffer();
}

// STEP 5: DOWNLOAD BUFFER FROM URL
//   - Download binary buffer from a remote asset url for next processing
async function downloadBuffer(url) {
  const maxAttempts = 4;
  const timeoutMs = 20000;
  let lastError;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(function() {
      controller.abort();
    }, timeoutMs);

    try {
      const res = await fetch(url, { signal: controller.signal });
      clearTimeout(timeout);

      if (!res.ok) {
        // Retry transient upstream failures; fail fast for stable client-side statuses.
        if (res.status >= 500 && attempt < maxAttempts) {
          await new Promise(function(resolve) { setTimeout(resolve, 300 * attempt); });
          continue;
        }
        throw new Error("Failed to download " + url + " (" + res.status + ")");
      }

      const ab = await res.arrayBuffer();
      return Buffer.from(ab);
    } catch (error) {
      clearTimeout(timeout);
      lastError = error;
      if (attempt < maxAttempts) {
        await new Promise(function(resolve) { setTimeout(resolve, 300 * attempt); });
        continue;
      }
    }
  }

  const msg =
    (lastError && lastError.message)
      ? String(lastError.message)
      : "unknown fetch error";
  throw new Error("Failed to download after retries: " + url + " (" + msg + ")");
}

// STEP 0: START SERVER - bind to port 3000
app.listen(3000, function() {
  console.log("Server is running on port 3000");
});
