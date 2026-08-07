# Supabase setup

## 1. Environment variables

The included `.env` already contains this project's public Supabase URL,
publishable key, and JWKS URL. Fill in only:

- `OPENAI_API_KEY`
- `FLASK_SECRET_KEY`
- `APP_BASE_URL` (use `http://127.0.0.1:5000` locally)
- `DAILY_AI_LIMIT` (optional; defaults to 200)

Generate a Flask secret with:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

The app does not require a Supabase secret/service-role key for authentication
or design CRUD. It uses the publishable key and each user's session so that
Row Level Security remains active.

## 2. Database schema

In the Supabase dashboard, open **SQL Editor**, paste the contents of
`supabase_schema.sql`, and run it once.

The schema includes profiles, saved designs, version history, RLS policies,
atomic three-design/version functions, database-backed AI usage quotas, free-trial
entitlements (three generations and ten AI edits), provider token telemetry, and
private-beta feedback submissions.

## 3. Authentication redirect URLs

In **Authentication → URL Configuration**, add:

- `http://127.0.0.1:5000/?reset=1`
- `http://localhost:5000/?reset=1`

Add the deployed equivalent later. Password recovery cannot complete unless
Supabase permits the redirect URL.

## 4. Install and run

```bash
cd ~/flowchart-app
pip3 install -r requirements.txt
python3 app.py
```

Then open `http://localhost:5000`.

## 5. Security

- Never commit `.env`.
- Never expose a secret/service-role key in browser JavaScript.
- Rotate any secret key that has been pasted into a chat, ticket, screenshot,
  or public repository.

## Updating an existing Supabase project

Run the full `supabase_schema.sql` again after installing this build. The script is
idempotent and adds the latest entitlement, usage, and Row Level Security changes.
