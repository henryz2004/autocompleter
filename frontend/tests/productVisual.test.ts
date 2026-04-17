import { beforeEach, describe, expect, it } from "vitest";

import { initProductVisual } from "../src/lib/productVisual";

function buildMarkup(): HTMLElement {
  const root = document.createElement("div");
  root.innerHTML = `
    <div data-draft-target>that I keep building what I think they need, not what they ask for</div>
    <ol>
      <li
        class="overlay-item is-active"
        tabindex="0"
        role="button"
        aria-pressed="true"
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
  });

  it("updates the draft when a suggestion is clicked", () => {
    const root = buildMarkup();
    initProductVisual({ root });

    const suggestion = root.querySelectorAll<HTMLElement>("[data-suggestion-item]")[1]!;
    suggestion.click();

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

    expect(root.querySelector<HTMLElement>("[data-draft-target]")?.textContent).toBe(
      "that we haven't found one person who couldn't live without it",
    );
  });
});
