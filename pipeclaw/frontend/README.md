# PipeClaw frontend

This is the Vite + React interface for exploring pipeline flows and chatting
with the PipeClaw agent. It is a source-only frontend; the FastAPI backend must
be running separately.

## Run locally

From this directory:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. The development server listens on all interfaces
and proxies `/api` and `/assets` to `http://localhost:8003` (see
`vite.config.ts`). Start the backend first:

```bash
python -m pipeclaw.backend.main
```

## Useful scripts

- `npm run dev` — start the Vite development server.
- `npm run build` — run TypeScript checks and create a production build.
- `npm run preview` — preview the production build locally.
- `npm run lint` — run ESLint with warnings treated as errors.

The main screens and API clients live under `src/`; flow data is fetched from
the backend rather than bundled into the frontend.
