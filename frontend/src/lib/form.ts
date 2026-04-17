/**
 * Wires up the waitlist form element. Pure DOM glue so it can be initialized
 * from an Astro `<script>` tag AND unit-tested with happy-dom.
 */

import {
  submitBetaApplication,
  type BetaApplicationSuccess,
} from "./api";
import {
  hasErrors,
  validateWaitlistForm,
  type FieldErrors,
} from "./validation";

export interface InitFormOptions {
  root: HTMLElement;
  baseUrl: string;
}

export function initWaitlistForm({ root, baseUrl }: InitFormOptions): void {
  const form = root.querySelector<HTMLFormElement>("[data-form]");
  const formContainer = root.querySelector<HTMLElement>("[data-form-container]");
  const successContainer = root.querySelector<HTMLElement>(
    "[data-success-container]",
  );
  const submitButton = root.querySelector<HTMLButtonElement>("[data-submit]");
  const errorBanner = root.querySelector<HTMLElement>("[data-error-banner]");

  if (
    !form ||
    !formContainer ||
    !successContainer ||
    !submitButton ||
    !errorBanner
  ) {
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearFieldErrors(root);
    setBannerMessage(errorBanner, "");

    const data = readFormData(form);
    const errors = validateWaitlistForm(data);

    if (hasErrors(errors)) {
      applyFieldErrors(root, errors);
      return;
    }

    setLoading(submitButton, true);

    const honeypot =
      form.querySelector<HTMLInputElement>("[data-honeypot]")?.value ?? "";

    const result = await submitBetaApplication(data, {
      baseUrl,
      honeypot,
      source: "landing",
    });

    setLoading(submitButton, false);

    if (result.ok) {
      renderSuccess(successContainer, result.value);
      formContainer.hidden = true;
      successContainer.hidden = false;
      successContainer.classList.add("success-enter");
      successContainer.focus({ preventScroll: false });
      successContainer.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }

    setBannerMessage(errorBanner, result.message);
  });
}

function readFormData(form: HTMLFormElement) {
  const getValue = (name: string) =>
    ((form.elements.namedItem(name) as HTMLInputElement | HTMLTextAreaElement | null)
      ?.value ?? "");
  return {
    name: getValue("name"),
    email: getValue("email"),
    role: getValue("role"),
    primary_use_case: getValue("primary_use_case"),
  };
}

function clearFieldErrors(root: HTMLElement) {
  root.querySelectorAll<HTMLElement>("[data-field-error]").forEach((el) => {
    el.textContent = "";
    el.hidden = true;
  });
  root.querySelectorAll<HTMLElement>("[data-field]").forEach((el) => {
    el.removeAttribute("data-invalid");
  });
}

function applyFieldErrors(root: HTMLElement, errors: FieldErrors) {
  (Object.entries(errors) as [keyof FieldErrors, string][]).forEach(
    ([field, message]) => {
      const wrapper = root.querySelector<HTMLElement>(
        `[data-field="${field}"]`,
      );
      const errorEl = root.querySelector<HTMLElement>(
        `[data-field-error="${field}"]`,
      );
      if (wrapper) wrapper.setAttribute("data-invalid", "true");
      if (errorEl) {
        errorEl.textContent = message;
        errorEl.hidden = false;
      }
    },
  );
  const first = root.querySelector<HTMLElement>("[data-invalid] input, [data-invalid] textarea");
  first?.focus();
}

function setLoading(button: HTMLButtonElement, loading: boolean) {
  button.disabled = loading;
  button.setAttribute("aria-busy", loading ? "true" : "false");
  button.dataset.loading = loading ? "true" : "false";
  const label = button.querySelector<HTMLElement>("[data-label]");
  const spinner = button.querySelector<HTMLElement>("[data-spinner]");
  if (label) label.hidden = loading;
  if (spinner) spinner.hidden = !loading;
}

function setBannerMessage(banner: HTMLElement, message: string) {
  banner.textContent = message;
  banner.hidden = !message;
}

export function renderSuccess(
  container: HTMLElement,
  data: BetaApplicationSuccess,
): void {
  const envSetupEl = container.querySelector<HTMLElement>("[data-env-setup]");
  const docsLinkEl = container.querySelector<HTMLAnchorElement>("[data-docs-link]");

  if (envSetupEl) envSetupEl.textContent = data.env_setup;
  if (docsLinkEl && data.install_docs_url) {
    docsLinkEl.href = data.install_docs_url;
  }

  container
    .querySelectorAll<HTMLButtonElement>("[data-copy-target]")
    .forEach((button) => {
      button.addEventListener("click", async () => {
        const targetSelector = button.dataset.copyTarget;
        if (!targetSelector) return;
        const target = container.querySelector<HTMLElement>(targetSelector);
        if (!target || !target.textContent) return;
        try {
          await navigator.clipboard?.writeText(target.textContent);
          const originalLabel = button.textContent ?? "Copy";
          button.textContent = "Copied";
          setTimeout(() => {
            button.textContent = originalLabel;
          }, 1200);
        } catch {
          // Clipboard unavailable; no-op. The user can still select the text.
        }
      });
    });
}
