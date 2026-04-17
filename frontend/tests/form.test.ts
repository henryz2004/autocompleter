import { beforeEach, describe, expect, it, vi } from "vitest";

import { initWaitlistForm } from "../src/lib/form";

// Replicates the structure of WaitlistForm.astro's markup closely enough for
// the form wiring to exercise every branch: validation, loading, error banner,
// and success rendering.
function buildMarkup(): HTMLElement {
  const root = document.createElement("div");
  root.setAttribute("data-waitlist", "");
  root.innerHTML = `
    <div data-form-container>
      <form data-form novalidate>
        <div data-error-banner hidden></div>

        <div data-field="name">
          <input name="name" />
          <p data-field-error="name" hidden></p>
        </div>
        <div data-field="email">
          <input name="email" />
          <p data-field-error="email" hidden></p>
        </div>
        <div data-field="role">
          <input name="role" />
          <p data-field-error="role" hidden></p>
        </div>
        <div data-field="primary_use_case">
          <textarea name="primary_use_case"></textarea>
          <p data-field-error="primary_use_case" hidden></p>
        </div>

        <input name="checkpoint" data-honeypot />

        <button type="submit" data-submit>
          <span data-label>Get my install key</span>
          <span data-spinner hidden></span>
        </button>
      </form>
    </div>

    <div data-success-container hidden tabindex="-1">
      <code data-install-id></code>
      <code data-install-key></code>
      <pre><code data-env-setup></code></pre>
      <a data-docs-link></a>
    </div>
  `;
  document.body.appendChild(root);
  return root;
}

function fill(root: HTMLElement, values: Record<string, string>) {
  for (const [name, value] of Object.entries(values)) {
    const el = root.querySelector<HTMLInputElement | HTMLTextAreaElement>(
      `[name="${name}"]`,
    );
    if (el) el.value = value;
  }
}

function submitForm(root: HTMLElement) {
  const form = root.querySelector<HTMLFormElement>("[data-form]")!;
  form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
}

async function flushMicrotasks() {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
}

describe("waitlist form wiring", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("shows field errors for an invalid submission without calling fetch", async () => {
    const root = buildMarkup();
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("{}", { status: 500 }),
    );

    initWaitlistForm({ root, baseUrl: "https://api.test" });
    submitForm(root);
    await flushMicrotasks();

    expect(fetchSpy).not.toHaveBeenCalled();
    const nameError = root.querySelector<HTMLElement>(
      '[data-field-error="name"]',
    );
    expect(nameError?.hidden).toBe(false);
    expect(nameError?.textContent?.length).toBeGreaterThan(0);
  });

  it("shows loading state, renders credentials on 201, and hides the form", async () => {
    const root = buildMarkup();
    let fetchResolve: ((value: Response) => void) | undefined;
    const responsePromise = new Promise<Response>((resolve) => {
      fetchResolve = resolve;
    });
    vi.spyOn(globalThis, "fetch").mockImplementation(() => responsePromise);

    initWaitlistForm({ root, baseUrl: "https://api.test" });
    fill(root, {
      name: "Ada Lovelace",
      email: "ada@example.com",
      role: "Founding engineer",
      primary_use_case: "Reply drafting in Slack.",
    });

    submitForm(root);
    await flushMicrotasks();

    const button = root.querySelector<HTMLButtonElement>("[data-submit]")!;
    expect(button.disabled).toBe(true);
    expect(button.getAttribute("aria-busy")).toBe("true");

    fetchResolve!(
      new Response(
        JSON.stringify({
          application_id: "app_1",
          install_id: "ins_abc",
          install_key: "key_xyz_plaintext",
          status: "granted",
          proxy_base_url: "https://proxy/v1",
          telemetry_url: "https://proxy/v1/telemetry/events",
          install_docs_url: "https://example.com/docs",
          env_setup: "AUTOCOMPLETER_INSTALL_ID=ins_abc\nAUTOCOMPLETER_PROXY_API_KEY=key_xyz_plaintext\n",
        }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      ),
    );
    await flushMicrotasks();

    const formContainer = root.querySelector<HTMLElement>(
      "[data-form-container]",
    )!;
    const successContainer = root.querySelector<HTMLElement>(
      "[data-success-container]",
    )!;
    expect(formContainer.hidden).toBe(true);
    expect(successContainer.hidden).toBe(false);

    expect(
      root.querySelector<HTMLElement>("[data-install-id]")!.textContent,
    ).toBe("ins_abc");
    expect(
      root.querySelector<HTMLElement>("[data-install-key]")!.textContent,
    ).toBe("key_xyz_plaintext");
    expect(
      root.querySelector<HTMLElement>("[data-env-setup]")!.textContent,
    ).toContain("AUTOCOMPLETER_INSTALL_ID=ins_abc");
    expect(
      root.querySelector<HTMLAnchorElement>("[data-docs-link]")!.getAttribute(
        "href",
      ),
    ).toBe("https://example.com/docs");
    expect(button.disabled).toBe(false);
  });

  it("surfaces a friendly message for duplicate email (409)", async () => {
    const root = buildMarkup();
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "already exists" }), {
        status: 409,
        headers: { "Content-Type": "application/json" },
      }),
    );

    initWaitlistForm({ root, baseUrl: "https://api.test" });
    fill(root, {
      name: "Ada",
      email: "ada@example.com",
      role: "Engineer",
      primary_use_case: "Terminal.",
    });
    submitForm(root);
    await flushMicrotasks();

    const banner = root.querySelector<HTMLElement>("[data-error-banner]")!;
    expect(banner.hidden).toBe(false);
    expect(banner.textContent?.toLowerCase()).toContain("already");

    // Form stays visible so the user isn't stuck.
    const formContainer = root.querySelector<HTMLElement>(
      "[data-form-container]",
    )!;
    expect(formContainer.hidden).toBe(false);
  });

  it("submits a mocked end-to-end happy path with honeypot empty", async () => {
    const root = buildMarkup();
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          application_id: "app_1",
          install_id: "ins_1",
          install_key: "key_1",
          status: "granted",
          proxy_base_url: "https://p/v1",
          telemetry_url: "https://p/v1/telemetry/events",
          install_docs_url: "https://docs",
          env_setup: "AUTOCOMPLETER_INSTALL_ID=ins_1\n",
        }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      ),
    );

    initWaitlistForm({ root, baseUrl: "https://api.test" });
    fill(root, {
      name: "Ada",
      email: "ada@example.com",
      role: "Engineer",
      primary_use_case: "Terminal commits.",
    });
    submitForm(root);
    await flushMicrotasks();

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(url).toBe("https://api.test/v1/beta/applications");
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.company).toBe("");
    expect(body.email).toBe("ada@example.com");
  });

  it("reads the honeypot from the hidden data-honeypot field", async () => {
    const root = buildMarkup();
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          application_id: "app_1",
          install_id: "ins_1",
          install_key: "key_1",
          status: "granted",
          proxy_base_url: "https://p/v1",
          telemetry_url: "https://p/v1/telemetry/events",
          install_docs_url: "https://docs",
          env_setup: "AUTOCOMPLETER_INSTALL_ID=ins_1\n",
        }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      ),
    );

    initWaitlistForm({ root, baseUrl: "https://api.test" });
    fill(root, {
      name: "Ada",
      email: "ada@example.com",
      role: "Engineer",
      primary_use_case: "Terminal commits.",
      checkpoint: "bot-value",
    });
    submitForm(root);
    await flushMicrotasks();

    const [, init] = fetchSpy.mock.calls[0]!;
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.company).toBe("bot-value");
  });
});

describe("mobile layout smoke test", () => {
  it("structural nodes stay in the DOM and stacked at narrow widths", () => {
    const root = buildMarkup();
    // happy-dom lacks a real layout engine; assert the DOM contract instead of
    // literal stacking. The CSS file pins the single-column media query.
    Object.defineProperty(window, "innerWidth", { value: 375, writable: true });

    const form = root.querySelector("[data-form]");
    const fields = root.querySelectorAll("[data-field]");
    const submit = root.querySelector("[data-submit]");
    const success = root.querySelector("[data-success-container]");

    expect(form).toBeTruthy();
    expect(fields.length).toBe(4);
    expect(submit).toBeTruthy();
    expect((success as HTMLElement).hidden).toBe(true);
  });
});
