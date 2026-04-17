import { describe, expect, it, vi } from "vitest";

import { buildApplicationsUrl, submitBetaApplication } from "../src/lib/api";

const payload = {
  name: "Ada",
  email: "ada@example.com",
  role: "Engineer",
  primary_use_case: "Terminal commits.",
};

function mockResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("buildApplicationsUrl", () => {
  it("appends /v1/beta/applications to a bare base url", () => {
    expect(buildApplicationsUrl("https://api.example.com")).toBe(
      "https://api.example.com/v1/beta/applications",
    );
  });

  it("only appends /beta/applications when base already ends in /v1", () => {
    expect(buildApplicationsUrl("https://api.example.com/v1")).toBe(
      "https://api.example.com/v1/beta/applications",
    );
  });

  it("strips trailing slashes", () => {
    expect(buildApplicationsUrl("https://api.example.com///")).toBe(
      "https://api.example.com/v1/beta/applications",
    );
  });

  it("falls back to a relative path when base is empty", () => {
    expect(buildApplicationsUrl("")).toBe("/v1/beta/applications");
  });
});

describe("submitBetaApplication", () => {
  it("posts JSON to the applications URL", async () => {
    const fetchImpl = vi.fn(async () =>
      mockResponse(201, {
        application_id: "app_1",
        install_id: "ins_1",
        install_key: "key_xyz",
        status: "granted",
        proxy_base_url: "https://p/v1",
        telemetry_url: "https://p/v1/telemetry/events",
        install_docs_url: "https://docs",
        env_setup: "AUTOCOMPLETER_INSTALL_ID=ins_1\n",
      }),
    );

    const result = await submitBetaApplication(payload, {
      baseUrl: "https://api.example.com",
      honeypot: "",
      source: "landing",
      fetchImpl,
    });

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0]!;
    expect(url).toBe("https://api.example.com/v1/beta/applications");
    expect(init?.method).toBe("POST");
    expect((init?.headers as Record<string, string>)["Content-Type"]).toBe(
      "application/json",
    );

    const sentBody = JSON.parse(init?.body as string);
    expect(sentBody.email).toBe("ada@example.com");
    expect(sentBody.company).toBe("");
    expect(sentBody.source).toBe("landing");

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.install_key).toBe("key_xyz");
    }
  });

  it("maps 409 to a friendly duplicate-email message", async () => {
    const fetchImpl = vi.fn(async () =>
      mockResponse(409, { detail: "already exists" }),
    );
    const result = await submitBetaApplication(payload, {
      baseUrl: "https://api.example.com",
      fetchImpl,
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(409);
      expect(result.message.toLowerCase()).toContain("already");
    }
  });

  it("surfaces 422 details from the backend", async () => {
    const fetchImpl = vi.fn(async () =>
      mockResponse(422, { detail: "invalid email address" }),
    );
    const result = await submitBetaApplication(payload, {
      baseUrl: "https://api.example.com",
      fetchImpl,
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(422);
      expect(result.message).toContain("invalid email");
    }
  });

  it("handles fetch throwing as a network error", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new Error("offline");
    });
    const result = await submitBetaApplication(payload, {
      baseUrl: "https://api.example.com",
      fetchImpl,
    });
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.status).toBe(0);
      expect(result.message).toContain("offline");
    }
  });
});
