# Data Flows

This document gives a practical, high-level overview of how data is expected to move through `autocompleter`.

It is intentionally simple. The exact product behavior may evolve, but the key trust boundary should remain understandable.

## Core Local Flow

In the core local flow:

1. The app detects the focused input.
2. It reads local context near the cursor using Accessibility APIs and app-specific extractors.
3. It assembles a prompt from local context, mode detection, and optional local memory.
4. It sends that prompt to the configured model provider.
5. It renders returned suggestions locally in the overlay.
6. If the user accepts a suggestion, the app injects text back into the focused app.

The most important point is that the local client is responsible for:

- reading context
- deciding what context is included
- sending the request
- receiving and rendering the response

## Bring Your Own Provider

When using a bring-your-own provider setup:

- prompt data goes from the local app to the provider you configured
- Autocompleter does not need to be in the middle of that request path
- the provider's own policies and terms still apply

This is the simplest trust story and should remain a first-class path.

## Telemetry

The intended product direction is that telemetry is available and user-controllable, with a clear opt-out path.

Telemetry is intended to mean product and reliability signals such as:

- app version
- feature usage
- latency
- error types
- coarse suggestion interaction signals

Telemetry is not intended to mean routine collection of raw prompt content.

## Improvement Signals

Improvement signals are product-feedback signals such as:

- accepted suggestion
- dismissed suggestion
- partial accept
- coarse quality or ranking outcomes

These signals may be useful for evaluating and improving ranking or product behavior. They should be treated more carefully than ordinary operational counters, especially if they could become user-identifying at scale.

## Training Content

Training content is the highest-sensitivity category. This includes:

- raw prompt text
- raw extracted conversation or terminal context
- raw completions
- accepted text content
- memory content

The intended direction is that this category requires a stronger consent boundary than ordinary product operation.

## Optional Future Cloud Flows

If future cloud features are added, they may include flows such as:

- account and billing
- settings sync
- hosted memory sync
- team or enterprise shared context
- managed provider routing

Those flows should be disclosed separately from the core local flow, because they represent a broader data boundary than basic BYO-provider usage.

## Managed Routing

If Autocompleter ever offers managed routing or managed providers, the product should describe that plainly in user-facing terms:

- whether prompts are routed through Autocompleter infrastructure
- whether content is stored, transformed, or only proxied
- whether that path is required or optional

Users should not have to infer that from an abstract feature name.

## Research And Model Improvement

If research or model-improvement programs are offered, they should be clearly separate from ordinary usage.

The high-level boundary should be:

- ordinary usage does not imply unrestricted training use of user content
- any broader content use should be disclosed and consented to explicitly

## Design Principle

When in doubt, the product should survive this question:

> If a security researcher published exactly what this feature sends over the network, would users feel that the behavior matched the product's promise?

That is the standard this project should aim for as it grows.

