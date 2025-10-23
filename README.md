# Serie A Match Predictor — Frontend

This repository contains a simple responsive frontend for a Serie A match predictor. The frontend will try to POST to `/api/predict` (JSON) for predictions; if the backend is not available it uses a built-in heuristic fallback.

Files added:

- `Index.html` — main page
- `styles.css` — styling
- `main.js` — frontend logic and client-side fallback predictor

How to run locally:

1. Open `Index.html` in a browser. For full `fetch` behaviour you'll want to serve files with a local static server.

   Example using Python (powershell):

   ```powershell
   python -m http.server 8000
   # then open http://localhost:8000 in a browser
   ```

2. To wire your backend later, implement a POST `/api/predict` that accepts a JSON body with the form inputs and returns:

```json
{
  "probs": { "home": 0.55, "draw": 0.25, "away": 0.20 },
  "explanation": "Optional text explaining the model output"
}
```

Next steps you might want me to do:
- Add TypeScript and bundler (Vite) for a modern workflow
- Wire real team data and caching
- Replace heuristic with a client-side ML model (tfjs) for offline predictions
# Footballanalyzer
School project using ML to anallize and predict football match outcomes.

@Author Steffen Ulvestad, August Mareno Hansen

During our time at HVL we we're tasked on creating our own self defined ML project. We decided on creating a football match predictor.
