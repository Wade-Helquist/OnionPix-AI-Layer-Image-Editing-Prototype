export const DEFAULT_NEGATIVE_MASK_PROMPT = "";

export const LABEL_PROMPT = `Identify the single main subject/object in this crop.
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

export function labelNounRewritePrompt(label) {
  return `Rewrite this as a noun-only object/type label for segmentation masking.
Reason: this text drives mask isolation, so actions/background context harms mask quality.
Rules:
- Return plain text only (no JSON, no punctuation).
- Return only the object/type identity as a noun phrase.
- Remove actions, verbs, background/scene context, and effects.
- Keep it concise (2-6 words), lowercase.
Text: "${label}"`;
}

export function negativePrompt(label) {
  return `Main target label: "${label}"
From this crop, list other visible entities that are NOT the main target.
Include secondary people/objects/props/scene elements if visible.
Do NOT include clothing, fashion wearables, or body parts.
Keep weapons/tools/props if visible.
Return JSON array only. Example: ["person behind", "chair", "window"]`;
}

export const EXTEND_PROFILE_GOLDEN = "A perfectly square 1:1 standalone asset on a pure white background. Based on the provided cutout, centrally frame a whole, seamless, fully complete version of the same subject with massive negative space around it. No part of the subject touches the canvas edges. The subject floats in unsupported levitation in a pure white void, attached to absolutely nothing. Remove all stands, bases, perches, ground surfaces, and support elements visible or implied by the cutout. Fully extend and complete all previously cut-off, hidden, missing, or obscured parts from the cutout into finished anatomical or structural form. Keep the exact material texture, color palette, and lighting style from the cutout while ignoring any local artifact contamination along its edges. No new objects, props, effects, text, or background content. Focus only on the subject's pure intact form as derived solely from the cutout.";

export function extendPrompt(subjectDescription = "") {
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
}

export function fullRemovePrompt(label, rect) {
  return `Remove only the ${label} near this region: x=${rect.x}, y=${rect.y}, width=${rect.width}, height=${rect.height}.
Keep every other person, object, and the background unchanged.
Do not remove anything else.
Fill the removed area naturally to match the surrounding scene and lighting.
Output the same full image dimensions with only that target removed.`;
}

export const BG_CLEANUP_PROMPT =
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

export function swapSubjectPrompt(rect, originalLabel, instructionText, footprintHint) {
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
}

export function extendDecisionPrompt(label) {
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
}

export function cutoutDescriptionPrompt(label) {
  return `You are given an isolated subject cutout for "${label}".
Describe the main subject based only on the isolated cutout image.
Rules:
- 8-20 words, plain English.
- Focus only on visible subject form/material/features.
- No background, scene, effects, or extra objects.
- Keep identity aligned to "${label}".
Return JSON only:
{"description":"short plain description of the isolated subject"}`;
}

