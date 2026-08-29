# V-Link Paper README Design

## Goal

Replace the inherited project README with an original paper-facing README for
“V-Link: Recovering Lost Visual Representations in Action DiT for
Vision-Language-Action Models.”

## Identity and scope

- Present V-Link as research code by the paper authors.
- Do not retain the inherited banner, product copy, checkpoint catalog, generic
  examples, or upstream project navigation.
- Mention GR00T N1.6 only as the initialization base and software dependency.
- State explicitly that this public release currently provides the fixed
  RoboTwin2.0 reproduction path; do not imply that LIBERO or real-robot training
  code is included.

## Structure

1. Paper title, authors, affiliations, and a link to the repository.
2. A concise summary of the visual-representation accessibility problem and
   V-Link's Spatial/Semantic Query solution.
3. Paper results, clearly labeled as reported results rather than promises of
   released benchmark code.
4. Released-code scope and fixed RoboTwin2.0 configuration.
5. Environment setup, base checkpoint, dataset validation, Warmup, Full,
   serving, and evaluation commands.
6. Acknowledgements, license, and citation.

## Reproduction contract

- Use the existing `gr00t_n1.6` environment for model training and serving.
- Follow the official RoboTwin repository for simulator setup.
- Download GR00T-N1.6-3B from its official Hugging Face repository and point
  `BASE_MODEL_PATH` to the resulting local directory.
- Link detailed dataset schema and operational overrides to
  `examples/RoboTwin2/README.md` instead of duplicating all details.

## Verification

- Check that inherited project branding and navigation are absent.
- Check every command and fixed hyperparameter against the retained RoboTwin2
  scripts and recipe tests.
- Run the focused RoboTwin2 recipe tests and shell syntax checks before push.
