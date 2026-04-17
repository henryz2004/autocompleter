create table if not exists public.beta_applications (
    application_id text primary key,
    email text not null,
    email_normalized text not null unique,
    name text not null,
    role text not null,
    primary_use_case text not null,
    status text not null default 'granted' check (
        status in ('granted', 'pending', 'rejected')
    ),
    install_id text references public.beta_installs (install_id),
    source text,
    submitted_at timestamptz not null default timezone('utc', now()),
    granted_at timestamptz
);

alter table public.beta_applications enable row level security;

create index if not exists beta_applications_email_normalized_idx
    on public.beta_applications (email_normalized);
create index if not exists beta_applications_submitted_at_idx
    on public.beta_applications (submitted_at desc);
create index if not exists beta_applications_install_id_idx
    on public.beta_applications (install_id);
