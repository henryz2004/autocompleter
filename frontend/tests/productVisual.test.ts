import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { initProductVisual } from "../src/lib/productVisual";

function buildMarkup(): HTMLElement {
  const root = document.createElement("div");
  root.innerHTML = `
    <div data-draft-target></div>
    <ol>
      <li
        class="overlay-item"
        tabindex="0"
        role="button"
        aria-pressed="false"
        data-suggestion-item
        data-suggestion-text="that I keep building what I think they need, not what they ask for"
      >one</li>
      <li
        class="overlay-item"
        tabindex="0"
        role="button"
        aria-pressed="false"
        data-suggestion-item
        data-suggestion-text="that I'm not sure I actually want to run a company"
      >two</li>
      <li
        class="overlay-item"
        tabindex="0"
        role="button"
        aria-pressed="false"
        data-suggestion-item
        data-suggestion-text="that we haven't found one person who couldn't live without it"
      >three</li>
    </ol>
  `;
  document.body.appendChild(root);
  return root;
}

describe("product visual interactions", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts with an empty draft", () => {
    const root = buildMarkup();
    initProductVisual({ root });

    expect(root.querySelector<HTMLElement>("[data-draft-target]")?.textContent).toBe("");
    root.querySelectorAll<HTMLElement>("[data-suggestion-item]").forEach((suggestion) => {
      expect(suggestion.getAttribute("aria-pressed")).toBe("false");
      expect(suggestion.classList.contains("is-active")).toBe(false);
    });
  });

  it("types the selected suggestion into the draft", () => {
    const root = buildMarkup();
    initProductVisual({ root });

    const suggestion = root.querySelectorAll<HTMLElement>("[data-suggestion-item]")[1]!;
    suggestion.click();

    vi.advanceTimersByTime(48);
    expect(root.querySelector<HTMLElement>("[data-draft-target]")?.textContent).toBe("tha");

    vi.runAllTimers();
    expect(root.querySelector<HTMLElement>("[data-draft-target]")?.textContent).toBe(
      "that I'm not sure I actually want to run a company",
    );
    expect(suggestion.classList.contains("is-active")).toBe(true);
    expect(suggestion.getAttribute("aria-pressed")).toBe("true");
  });

  it("supports keyboard selection", () => {
    const root = buildMarkup();
    initProductVisual({ root });

    const suggestion = root.querySelectorAll<HTMLElement>("[data-suggestion-item]")[2]!;
    suggestion.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));

    vi.runAllTimers();
    expect(root.querySelector<HTMLElement>("[data-draft-target]")?.textContent).toBe(
      "that we haven't found one person who couldn't live without it",
    );
  });

  it("restarts the typing animation when a different suggestion is chosen", () => {
    const root = buildMarkup();
    initProductVisual({ root });

    const suggestions = root.querySelectorAll<HTMLElement>("[data-suggestion-item]");
    suggestions[0]!.click();
    vi.advanceTimersByTime(64);

    suggestions[2]!.click();
    vi.advanceTimersByTime(32);
    expect(root.querySelector<HTMLElement>("[data-draft-target]")?.textContent).toBe("th");

    vi.runAllTimers();
    expect(root.querySelector<HTMLElement>("[data-draft-target]")?.textContent).toBe(
      "that we haven't found one person who couldn't live without it",
    );
    expect(suggestions[0]!.getAttribute("aria-pressed")).toBe("false");
    expect(suggestions[2]!.getAttribute("aria-pressed")).toBe("true");
  });
});
