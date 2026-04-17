export interface InitProductVisualOptions {
  root: HTMLElement;
}

const TYPING_DELAY_MS = 16;

export function initProductVisual({ root }: InitProductVisualOptions): void {
  const draftTarget = root.querySelector<HTMLElement>("[data-draft-target]");
  const overlayItems = Array.from(
    root.querySelectorAll<HTMLElement>("[data-suggestion-item]"),
  );

  if (!draftTarget || overlayItems.length === 0) {
    return;
  }

  let typingRun = 0;

  const animateDraft = (suggestion: string) => {
    typingRun += 1;
    const currentRun = typingRun;
    draftTarget.textContent = "";

    for (let index = 0; index < suggestion.length; index += 1) {
      window.setTimeout(() => {
        if (typingRun !== currentRun) {
          return;
        }

        draftTarget.textContent = suggestion.slice(0, index + 1);
      }, TYPING_DELAY_MS * (index + 1));
    }
  };

  const setActiveSuggestion = (item: HTMLElement) => {
    const suggestion = item.dataset.suggestionText ?? item.textContent ?? "";
    overlayItems.forEach((candidate) => {
      candidate.classList.toggle("is-active", candidate === item);
      candidate.setAttribute(
        "aria-pressed",
        candidate === item ? "true" : "false",
      );
    });
    animateDraft(suggestion);
  };

  overlayItems.forEach((item) => {
    item.addEventListener("click", () => setActiveSuggestion(item));
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setActiveSuggestion(item);
      }
    });
  });
}
