export interface InitProductVisualOptions {
  root: HTMLElement;
}

export function initProductVisual({ root }: InitProductVisualOptions): void {
  const draftTarget = root.querySelector<HTMLElement>("[data-draft-target]");
  const overlayItems = Array.from(
    root.querySelectorAll<HTMLElement>("[data-suggestion-item]"),
  );

  if (!draftTarget || overlayItems.length === 0) {
    return;
  }

  const setActiveSuggestion = (item: HTMLElement) => {
    const suggestion = item.dataset.suggestionText ?? item.textContent ?? "";
    overlayItems.forEach((candidate) => {
      candidate.classList.toggle("is-active", candidate === item);
      candidate.setAttribute(
        "aria-pressed",
        candidate === item ? "true" : "false",
      );
    });
    draftTarget.textContent = suggestion;
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
