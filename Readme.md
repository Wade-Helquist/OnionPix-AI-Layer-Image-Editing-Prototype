# OnionPix

### AI Layer-Based Image Editing Prototype

> **Archived Prototype:** The original source code for OnionPix was lost after the development environment became unavailable. This repository documents the project's design, architecture, implementation approach, technical decisions, and lessons learned.

> An experimental AI-assisted image editing application that uses canvas-based object selection and multi-stage AI workflows to provide precise control over object isolation, removal, and manipulation.

---

## Overview

OnionPix was created to solve one of the biggest usability challenges in AI image editing: communicating **exactly which object** should be edited and **exactly where** it should be placed.

Instead of relying solely on text prompts, OnionPix allows users to interact directly with an image using a simple HTML Canvas interface. Users draw one or two rectangles around objects and desired locations while the application coordinates multiple AI services to perform segmentation, masking, object removal, and image cleanup.

The project explored how thoughtful software architecture and user interface design could improve AI-assisted image editing by combining multiple specialized AI services into a single workflow.

---

# Why OnionPix?

The name **OnionPix** comes from the concept of peeling an image into layers.

Rather than treating an image as a single picture, OnionPix separates it into reusable components including:

- Original Image
- Isolated Objects
- AI Generated Masks
- Clean Backgrounds
- Future Replacement Objects

This layer-based approach was designed to make AI editing more modular, reusable, and easier to control.

---

# Motivation

Traditional AI image editors often struggle with precise user intent.

Instead of typing:

> "Move the dog slightly to the left."

The user simply draws:

- Rectangle A → Select Object
- Rectangle B → Destination or Target Area

This provides precise visual input while allowing AI models to perform the complex image processing.

---

# Features

- Canvas-based image editing
- Rectangle object selection
- Image cropping
- AI object segmentation
- Automatic mask generation
- Object isolation
- Object removal
- Automated background cleanup
- Intermediate workflow visualization
- Multi-stage AI processing pipeline

---

# Planned Editing Modes

- ✅ Remove
- 🚧 Move
- 🚧 Replace
- 🚧 Swap

---

# Processing Workflow

1. Upload an image.
2. Draw a rectangle around the target object.
3. Crop the selected region.
4. Send the crop to an AI segmentation service.
5. Receive and apply the generated object mask.
6. Isolate the selected object.
7. Remove the isolated object from the original image.
8. Send the remaining image to an AI image generation service for cleanup.
9. Display each intermediate processing stage for testing and debugging.

---

# What Worked Well

The prototype achieved consistently strong results for:

- Accurate object isolation
- AI-generated object masking
- Object removal
- Automated background cleanup
- Canvas interaction
- Rectangle selection workflow

One particularly successful workflow repeatedly detected remaining background objects, removed them, regenerated the image, and continued until no additional objects were detected, producing clean background images that could later be reused.

---

# Technical Challenges

The planned Move, Replace, and Swap operations proved to be limited by the capabilities of contemporary AI image generation models rather than the application's architecture.

The application successfully:

- Identified the selected object
- Isolated it
- Removed it
- Determined the intended destination

However, image generation models frequently:

- Changed object size
- Ignored the desired placement
- Altered unrelated portions of the scene
- Reinterpreted the selected object instead of preserving it

These limitations highlighted the challenges of maintaining spatial consistency in AI image generation during the prototype's development.

---

# My Contributions

This project was conceived, architected, and developed as an exploration into AI-assisted image editing workflows.

My contributions included:

- Designing the overall application architecture
- Creating the canvas-based editing interface
- Implementing rectangle selection and image cropping
- Designing the multi-stage editing workflow
- Integrating multiple AI APIs
- Orchestrating AI processing pipelines
- Building frontend interaction logic
- Testing and refining individual AI workflows
- Reading, modifying, and extending AI-assisted code while learning how each subsystem functioned
- Evaluating the practical limitations of emerging AI technologies

AI served as a collaborative development partner throughout the project, while I focused on the overall architecture, workflow design, frontend implementation, system integration, experimentation, and iterative refinement.

---

# Technologies

## Frontend

- HTML5
- JavaScript
- HTML Canvas

## AI Services

- Replicate
- Multiple AI Image Generation APIs
- REST APIs

## Concepts

- Object Segmentation
- Image Masking
- AI Workflow Orchestration
- Prompt Engineering
- Human–AI Collaboration

---

# Future Vision

The long-term vision extended beyond image editing into reusable AI-generated assets.

Planned future capabilities included:

- Searchable object library
- Searchable background library
- Cross-image object insertion
- Reusable extracted assets
- Collaborative image sharing
- Layer-based asset management

---

# Lessons Learned

This project reinforced several important software engineering principles:

- Effective AI applications require thoughtful workflow design, not just good prompts.
- User interface design can significantly improve AI usability.
- Breaking complex problems into multiple specialized AI tasks produces better results than relying on a single prompt.
- Iterative experimentation is essential when working with rapidly evolving AI technologies.
- Human-guided software architecture remains critical even when AI assists with implementation.

---

## Status

**Prototype / Archived Concept**

The original source code is no longer available after the development environment was lost. This repository documents the project's architecture, workflow, implementation approach, technical challenges, and lessons learned.
