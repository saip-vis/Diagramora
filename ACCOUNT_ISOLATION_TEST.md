# Two-account isolation test

Run this checklist in a private/incognito browser window after authentication or
session changes. Use two test accounts with clearly different design names.

## Normal ownership checks

1. Sign in as Account A and create, save, rename, duplicate, version, and export a design.
2. Copy Account A's design UUID from the browser network panel.
3. Log out, sign in as Account B, and confirm My Designs contains none of Account A's cards.
4. Request `/designs/<ACCOUNT_A_DESIGN_ID>` while signed in as Account B. Expect `404`.
5. Try the rename, delete, duplicate, and versions URLs with Account A's ID. Expect `404` and no changes.
6. Confirm Account B's generation/edit counters are independent from Account A's.

## In-flight account-switch checks

1. As Account A, begin an AI edit and immediately log out before it finishes.
2. Sign in as Account B. Confirm Account A's graph, title, editor fields, and dialogs never appear.
3. Repeat while an autosave is showing `Saving…`.
4. Repeat while My Designs or Versions is loading.
5. Confirm Account B did not receive a copied design and Account A's original was not modified.

## Expected database result

- Every `designs`, `design_versions`, `ai_entitlements`, `ai_usage_events`, and
  `feedback_submissions` row retains the correct `user_id`.
- Cross-account API requests return `404` rather than revealing whether the other row exists.
