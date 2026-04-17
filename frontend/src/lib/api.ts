import type { WaitlistFormInput } from "./validation";

export interface BetaApplicationSuccess {
  application_id: string;
  install_id: string;
  install_key: string;
  status: string;
  proxy_base_url: string;
  telemetry_url: string;
  install_docs_url: string;
  env_setup: string;
}

export type SubmitResult =
  | { ok: true; value: BetaApplicationSuccess }
  | { ok: false; status: number; message: string };

interface SubmitOptions {
  baseUrl: string;
  honeypot?: string;
  source?: string;
  // Injectable for tests. Falls back to globalThis.fetch.
  fetchImpl?: typeof fetch;
}

export async function submitBetaApplication(
  input: WaitlistFormInput,
  options: SubmitOptions,
): Promise<SubmitResult> {
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (!fetchImpl) {
    return { ok: false, status: 0, message: "fetch is not available" };
  }

  const url = buildApplicationsUrl(options.baseUrl);
  const body = {
    name: input.name.trim(),
    email: input.email.trim(),
    role: input.role.trim(),
    primary_use_case: input.primary_use_case.trim(),
    company: options.honeypot ?? "",
    source: options.source,
  };

  let response: Response;
  try {
    response = await fetchImpl(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Network error. Please try again.";
    return { ok: false, status: 0, message };
  }

  if (response.status === 201) {
    try {
      const data = (await response.json()) as BetaApplicationSuccess;
      return { ok: true, value: data };
    } catch {
      return {
        ok: false,
        status: response.status,
        message: "Server sent an unreadable response.",
      };
    }
  }

  if (response.status === 409) {
    return {
      ok: false,
      status: 409,
      message:
        "That email is already registered. If you've lost your install key, reach out and we'll mint a new one.",
    };
  }

  if (response.status === 422) {
    let detail = "Something about that submission was rejected.";
    try {
      const data = (await response.json()) as { detail?: string };
      if (data?.detail) detail = data.detail;
    } catch {
      // ignore JSON parse error; fall through
    }
    return { ok: false, status: 422, message: detail };
  }

  return {
    ok: false,
    status: response.status,
    message: "Something went wrong. Please try again in a moment.",
  };
}

export function buildApplicationsUrl(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, "");
  if (!trimmed) return "/v1/beta/applications";
  if (/\/v\d+$/.test(trimmed)) {
    return `${trimmed}/beta/applications`;
  }
  return `${trimmed}/v1/beta/applications`;
}
