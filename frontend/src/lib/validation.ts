/**
 * Pure validation helpers for the waitlist form.
 *
 * Matches the constraints enforced server-side in
 * `backend/app.py::BetaApplicationRequest`.
 */

export interface WaitlistFormInput {
  name: string;
  email: string;
  role: string;
  primary_use_case: string;
}

export type FieldErrors = Partial<Record<keyof WaitlistFormInput, string>>;

const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

export function validateWaitlistForm(input: WaitlistFormInput): FieldErrors {
  const errors: FieldErrors = {};
  const name = input.name.trim();
  const email = input.email.trim();
  const role = input.role.trim();
  const useCase = input.primary_use_case.trim();

  if (!name) errors.name = "Please enter your name.";
  else if (name.length > 200) errors.name = "Name is too long.";

  if (!email) errors.email = "Please enter your email.";
  else if (!EMAIL_RE.test(email)) errors.email = "That email doesn't look right.";
  else if (email.length > 320) errors.email = "Email is too long.";

  if (!role) errors.role = "Tell us what you do.";
  else if (role.length > 200) errors.role = "Keep this under 200 characters.";

  if (!useCase) errors.primary_use_case = "One sentence is enough.";
  else if (useCase.length > 2000)
    errors.primary_use_case = "Keep this under 2000 characters.";

  return errors;
}

export function hasErrors(errors: FieldErrors): boolean {
  return Object.keys(errors).length > 0;
}
