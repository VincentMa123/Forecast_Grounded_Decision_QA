# Repository Guidelines

## Project Structure & Module Organization
This repository currently contains source documents only. Keep project PDFs at the repository root unless a category grows enough to justify a subdirectory. Existing files are:

- `Forecast-Grounded_Decision_QA_for_Gas_Pipeline_Transient_Operation_EN.pdf` for the main gas-pipeline decision QA reference.
- `PipeFormer_PipeClaw_Schedule.pdf` for schedule and planning material.

Add future derived assets under descriptive folders such as `notes/`, `figures/`, or `scripts/` rather than mixing generated outputs with source PDFs.

## Build, Test, and Development Commands
No build system, package manager, or test runner is configured. Useful local checks are:

- `rg --files` lists repository files quickly.
- `Get-ChildItem` shows file sizes and modified dates in Windows PowerShell.

If scripts are added later, document their exact invocation here and keep generated outputs out of the root.

## Coding Style & Naming Conventions
There is no code style configuration yet. For documents, prefer descriptive filenames with project/topic, artifact type, and language when applicable, for example `Forecast-Grounded_Decision_QA_for_Gas_Pipeline_Transient_Operation_EN.pdf`. Avoid vague names like `final.pdf` or `new_version.pdf`. Keep file extensions lowercase.

## Testing Guidelines
No automated tests exist. For document updates, verify that PDFs open locally, page counts look expected, and filenames still match their contents. If code is introduced, add a colocated test directory and include the test command in this guide.

## Commit & Pull Request Guidelines
This folder is not currently initialized as a Git repository, so no local commit-message convention is available. If Git is introduced, use short imperative commits, for example `Add pipeline QA reference paper`, and keep PR descriptions focused on what changed, why, and any document version/date details.

## Agent-Specific Instructions
Before editing, inspect the current file list and avoid deleting or replacing source PDFs unless explicitly requested. Generated notes, OCR text, or converted images should be added as separate files with clear provenance.
