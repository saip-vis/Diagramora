-- Run this entire file in Supabase Dashboard -> SQL Editor.
create extension if not exists pgcrypto;

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  username text unique,
  display_name text,
  plan text not null default 'free' check (plan in ('free', 'pro', 'team')),
  trial_credits integer not null default 10 check (trial_credits >= 0),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.designs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null default 'Untitled Flowchart',
  workflow_json jsonb not null default '{}'::jsonb,
  thumbnail_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.billing_customers (
  user_id uuid primary key references auth.users(id) on delete cascade,
  stripe_customer_id text unique,
  stripe_subscription_id text unique,
  subscription_status text,
  current_period_end timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists designs_user_id_idx on public.designs(user_id);
alter table public.profiles enable row level security;
alter table public.designs enable row level security;
alter table public.billing_customers enable row level security;
grant select, insert, update, delete on public.profiles to authenticated;
grant select, update, delete on public.designs to authenticated;
revoke insert on public.designs from authenticated;
grant select on public.billing_customers to authenticated;

drop policy if exists "Users can view their own profile" on public.profiles;
drop policy if exists "Users can insert their own profile" on public.profiles;
drop policy if exists "Users can update their own profile" on public.profiles;
drop policy if exists "Users can view their own designs" on public.designs;
drop policy if exists "Users can create their own designs" on public.designs;
drop policy if exists "Users can update their own designs" on public.designs;
drop policy if exists "Users can delete their own designs" on public.designs;
drop policy if exists "Users can view their own billing record" on public.billing_customers;

create policy "Users can view their own profile" on public.profiles for select to authenticated using ((select auth.uid()) = id);
create policy "Users can insert their own profile" on public.profiles for insert to authenticated with check ((select auth.uid()) = id);
create policy "Users can update their own profile" on public.profiles for update to authenticated using ((select auth.uid()) = id) with check ((select auth.uid()) = id);
create policy "Users can view their own designs" on public.designs for select to authenticated using ((select auth.uid()) = user_id);
create policy "Users can create their own designs" on public.designs for insert to authenticated with check ((select auth.uid()) = user_id);
create policy "Users can update their own designs" on public.designs for update to authenticated using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
create policy "Users can delete their own designs" on public.designs for delete to authenticated using ((select auth.uid()) = user_id);
create policy "Users can view their own billing record" on public.billing_customers for select to authenticated using ((select auth.uid()) = user_id);

create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
declare
  requested_username text;
begin
  requested_username := coalesce(nullif(trim(new.raw_user_meta_data ->> 'username'), ''), split_part(new.email, '@', 1));
  if exists (select 1 from public.profiles where username = requested_username) then
    requested_username := left(requested_username, 50) || '-' || left(new.id::text, 8);
  end if;
  insert into public.profiles (id, username, display_name)
  values (
    new.id,
    requested_username,
    coalesce(new.raw_user_meta_data ->> 'display_name', split_part(new.email, '@', 1))
  ) on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created after insert on auth.users for each row execute procedure public.handle_new_user();

-- Version history for saved designs.
create table if not exists public.design_versions (
  id uuid primary key default gen_random_uuid(),
  design_id uuid not null references public.designs(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  version_number integer not null,
  workflow_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (design_id, version_number)
);

create index if not exists design_versions_design_id_idx on public.design_versions(design_id);
alter table public.design_versions enable row level security;
grant select, delete on public.design_versions to authenticated;
revoke insert on public.design_versions from authenticated;

drop policy if exists "Users can view their own design versions" on public.design_versions;
drop policy if exists "Users can create their own design versions" on public.design_versions;
drop policy if exists "Users can delete their own design versions" on public.design_versions;

create policy "Users can view their own design versions"
on public.design_versions for select to authenticated
using ((select auth.uid()) = user_id);

create policy "Users can create their own design versions"
on public.design_versions for insert to authenticated
with check ((select auth.uid()) = user_id);

create policy "Users can delete their own design versions"
on public.design_versions for delete to authenticated
using ((select auth.uid()) = user_id);

-- Server-enforced AI usage records. Users cannot write these rows directly;
-- the quota function records usage atomically using the caller's auth.uid().
create table if not exists public.ai_usage_events (
  id bigint generated by default as identity primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  operation text not null,
  created_at timestamptz not null default now()
);

create index if not exists ai_usage_events_user_operation_created_idx
on public.ai_usage_events(user_id, operation, created_at desc);

alter table public.ai_usage_events enable row level security;
revoke all on public.ai_usage_events from anon, authenticated;

create or replace function public.consume_ai_quota(
  p_operation text,
  p_window_seconds integer,
  p_limit integer
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  caller_id uuid := auth.uid();
  used_count integer;
begin
  if caller_id is null then
    return false;
  end if;
  if p_operation not in ('ai_total', 'generate', 'live_update', 'live_title', 'finalize', 'ai_edit', 'export')
     or p_window_seconds < 60 or p_window_seconds > 86400
     or p_limit < 1 or p_limit > 500 then
    raise exception 'Invalid quota configuration';
  end if;

  perform pg_advisory_xact_lock(hashtextextended(caller_id::text || ':' || p_operation, 0));
  select count(*) into used_count
  from public.ai_usage_events
  where user_id = caller_id
    and operation = p_operation
    and created_at >= now() - make_interval(secs => p_window_seconds);

  if used_count >= p_limit then
    return false;
  end if;

  insert into public.ai_usage_events(user_id, operation) values (caller_id, p_operation);
  delete from public.ai_usage_events
  where user_id = caller_id and created_at < now() - interval '2 days';
  return true;
end;
$$;

revoke all on function public.consume_ai_quota(text, integer, integer) from public;
grant execute on function public.consume_ai_quota(text, integer, integer) to authenticated;

-- Atomic three-design creation. The database, rather than the browser, owns
-- the test limit so concurrent requests cannot create a fourth design.
create or replace function public.create_design_limited(p_title text, p_workflow_json jsonb)
returns public.designs
language plpgsql
security definer
set search_path = public
as $$
declare
  caller_id uuid := auth.uid();
  created_design public.designs;
begin
  if caller_id is null then raise exception 'Authentication required'; end if;
  perform pg_advisory_xact_lock(hashtextextended(caller_id::text || ':designs', 0));
  if (select count(*) from public.designs where user_id = caller_id) >= 3 then
    raise exception using errcode = 'P0001', message = 'design_limit_reached';
  end if;
  insert into public.designs(user_id, title, workflow_json, updated_at)
  values (caller_id, left(coalesce(nullif(trim(p_title), ''), 'Untitled Flowchart'), 200), p_workflow_json, now())
  returning * into created_design;
  return created_design;
end;
$$;

create or replace function public.duplicate_design_limited(p_design_id uuid)
returns public.designs
language plpgsql
security definer
set search_path = public
as $$
declare
  caller_id uuid := auth.uid();
  source_design public.designs;
  created_design public.designs;
begin
  if caller_id is null then raise exception 'Authentication required'; end if;
  perform pg_advisory_xact_lock(hashtextextended(caller_id::text || ':designs', 0));
  if (select count(*) from public.designs where user_id = caller_id) >= 3 then
    raise exception using errcode = 'P0001', message = 'design_limit_reached';
  end if;
  select * into source_design from public.designs where id = p_design_id and user_id = caller_id;
  if source_design.id is null then raise exception using errcode = 'P0002', message = 'design_not_found'; end if;
  insert into public.designs(user_id, title, workflow_json, updated_at)
  values (caller_id, left(source_design.title || ' Copy', 200), source_design.workflow_json, now())
  returning * into created_design;
  return created_design;
end;
$$;

create or replace function public.create_design_version(p_design_id uuid)
returns public.design_versions
language plpgsql
security definer
set search_path = public
as $$
declare
  caller_id uuid := auth.uid();
  source_workflow jsonb;
  next_number integer;
  created_version public.design_versions;
begin
  if caller_id is null then raise exception 'Authentication required'; end if;
  perform pg_advisory_xact_lock(hashtextextended(caller_id::text || ':' || p_design_id::text || ':versions', 0));
  select workflow_json into source_workflow from public.designs where id = p_design_id and user_id = caller_id;
  if source_workflow is null then raise exception using errcode = 'P0002', message = 'design_not_found'; end if;
  select coalesce(max(version_number), 0) + 1 into next_number
  from public.design_versions where design_id = p_design_id and user_id = caller_id;
  insert into public.design_versions(design_id, user_id, version_number, workflow_json)
  values (p_design_id, caller_id, next_number, source_workflow)
  returning * into created_version;
  return created_version;
end;
$$;

revoke all on function public.create_design_limited(text, jsonb) from public;
revoke all on function public.duplicate_design_limited(uuid) from public;
revoke all on function public.create_design_version(uuid) from public;
grant execute on function public.create_design_limited(text, jsonb) to authenticated;
grant execute on function public.duplicate_design_limited(uuid) to authenticated;
grant execute on function public.create_design_version(uuid) to authenticated;
