import { describe, expect, it } from "vitest";

import { hasErrors, validateWaitlistForm } from "../src/lib/validation";

const valid = {
  name: "Ada Lovelace",
  email: "ada@example.com",
  role: "Founding engineer",
  primary_use_case: "Reply drafting in Slack.",
};

describe("validateWaitlistForm", () => {
  it("accepts a fully valid payload", () => {
    const errors = validateWaitlistForm(valid);
    expect(hasErrors(errors)).toBe(false);
  });

  it("rejects an empty name", () => {
    const errors = validateWaitlistForm({ ...valid, name: "   " });
    expect(errors.name).toBeDefined();
  });

  it("rejects malformed emails", () => {
    for (const bad of ["", "no-at-symbol", "a@b", "a @ b.com"]) {
      const errors = validateWaitlistForm({ ...valid, email: bad });
      expect(errors.email).toBeDefined();
    }
  });

  it("accepts common real-world emails", () => {
    for (const good of [
      "a.b@c.io",
      "Mixed@Case.CO.UK",
      "plus+tag@example.dev",
    ]) {
      const errors = validateWaitlistForm({ ...valid, email: good });
      expect(errors.email).toBeUndefined();
    }
  });

  it("rejects an empty role", () => {
    const errors = validateWaitlistForm({ ...valid, role: "" });
    expect(errors.role).toBeDefined();
  });

  it("rejects empty primary_use_case", () => {
    const errors = validateWaitlistForm({ ...valid, primary_use_case: "" });
    expect(errors.primary_use_case).toBeDefined();
  });

  it("rejects overlong fields", () => {
    const long = "x".repeat(3000);
    const errors = validateWaitlistForm({ ...valid, primary_use_case: long });
    expect(errors.primary_use_case).toBeDefined();
  });
});
