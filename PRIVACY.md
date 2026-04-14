# Privacy

`autocompleter` is a tool that operates close to sensitive user data. It can observe the currently focused input, read nearby Accessibility tree context, and send prompt data to a configured language model provider. This document explains the intended privacy posture of the project at a high level.

This is a product-direction document, not a final legal policy. The details here may evolve as the project matures.

## Privacy Posture

The project is intended to be:

- local-first by default
- inspectable in the code paths that handle local context
- usable with your own model provider credentials
- clear about when data leaves the device

The trust surface for this project is the local client: what it reads, how it assembles context, and where it sends that context.

## What The App Can Read

Depending on the active app and focused input, `autocompleter` may read:

- text before and after the cursor
- nearby Accessibility subtree context
- app name and window title
- browser URL when available
- conversation context extracted from supported chat apps
- terminal or TUI context when the focused app is a shell or terminal

This context is used to generate suggestions for the currently focused input.

## What Leaves The Device

At a minimum, prompt data leaves the device when the app sends a request to the configured LLM provider.

That provider may be:

- your own hosted model
- a third-party API you configure directly
- a future Autocompleter-managed service, if you choose to enable one

If you use a third-party or bring-your-own provider, your use of that provider is also governed by that provider's terms and privacy practices.

## Defaults We Intend To Ship

The intended default posture is:

- local-first usage supported without an Autocompleter cloud account
- telemetry available but user-controllable and opt-out
- content sharing for training disabled by default
- cloud sync and hosted memory optional, not required

Any future setting that increases data exposure beyond the normal request path should be clearly disclosed in-product.

## Data Categories

We think about data in four broad categories:

- local context
  Text and metadata used to generate a suggestion in the moment.
- operational telemetry
  Reliability and product-usage signals such as feature usage, latency, and error types.
- improvement signals
  Signals such as accept, dismiss, partial accept, and related product feedback.
- training content
  Raw or lightly transformed prompt/completion data that could be used to improve future models.

These categories should not be treated the same. In particular, training content should have a higher consent bar than basic product operation.

## Product Modes

The product direction currently assumes these broad modes:

- `Local-only`
  Use your own provider and keep product data on-device except for the provider you choose.
- `Cloud Sync`
  Add account-backed convenience features such as settings sync or multi-device continuity.
- `Research Opt-In`
  Separately allow selected data to be used to improve future models or ranking systems.
- `Enterprise`
  Support stronger administrative controls and a no-training-on-customer-content posture.

These are product-direction modes, not a final implementation contract.

## Local Storage

The app stores local state under `~/.autocompleter`. Depending on enabled features, this may include:

- local SQLite data
- suggestion feedback
- latency metrics
- optional local memory state
- optional debug dumps when explicitly enabled

Debugging artifacts such as `--dump-dir` output can contain sensitive context and should be handled carefully.

## What We Want To Avoid

We do not want the trust model for this product to depend on quietly retaining large amounts of raw user content.

The intended moat is product quality, infrastructure, distribution, and optional paid services, not surprise data collection.

## Future Work

More detailed documentation may be added later for:

- retention and deletion behavior
- consent revocation behavior
- cloud sync behavior
- enterprise controls
- encryption and storage details
- telemetry schemas and redaction rules

For now, the goal is to state the direction clearly: local-first, explicit about network boundaries, and conservative about content use.

