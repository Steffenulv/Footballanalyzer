
const API_BASE_URL = 'http://127.0.0.1:8000/api';


const homeTeamSelect = document.getElementById('home-team');
const awayTeamSelect = document.getElementById('away-team');
const predictionResult = document.getElementById('prediction-result');
const predictBtn = document.getElementById('predict-btn');


const teamStatsCache = new Map();
const controllers = { home: null, away: null };
async function safeJson(res) {
  try { return await res.json(); } catch { return null; }
}
function prettyLabel(v) {
  return String(v).replace(/\u00a0/g, ' ').trim();
}

// LOAD TEAMS
async function loadTeams() {
  try {
    const res = await fetch(`${API_BASE_URL}/teams`);
    if (!res.ok) throw new Error(`Failed to load teams (${res.status})`);
    const data = await res.json();
    const teams = Array.isArray(data) ? data : data?.teams;
    if (!Array.isArray(teams)) throw new Error('Invalid teams data format');

    homeTeamSelect.innerHTML = '<option value="">Select home team...</option>';
    awayTeamSelect.innerHTML = '<option value="">Select away team...</option>';

    teams.forEach(team => {
      const optionHome = document.createElement('option');
      optionHome.value = team;
      optionHome.textContent = prettyLabel(team);
      homeTeamSelect.appendChild(optionHome);

      const optionAway = document.createElement('option');
      optionAway.value = team;
      optionAway.textContent = prettyLabel(team);
      awayTeamSelect.appendChild(optionAway);
    });
  } catch (error) {
    console.error('Error loading teams:', error);
    alert('Failed to load teams.');
  }
}

//  LOAD STATS 
async function loadTeamStats(team, prefix) {
  try {
    if (teamStatsCache.has(team)) {
      updateStatsDisplay(teamStatsCache.get(team), prefix);
      return teamStatsCache.get(team);
    }
    if (controllers[prefix]) controllers[prefix].abort();
    controllers[prefix] = new AbortController();

    const res = await fetch(`${API_BASE_URL}/team/${encodeURIComponent(team)}`, {
      signal: controllers[prefix].signal
    });
    if (!res.ok) {
      const body = await safeJson(res);
      throw new Error(body?.detail || `HTTP ${res.status}`);
    }
    const stats = await res.json();
    teamStatsCache.set(team, stats);
    updateStatsDisplay(stats, prefix);
    return stats;
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.error(`Failed to load ${team} stats:`, err);
      clearStatsDisplay(prefix);
    }
  } finally {
    controllers[prefix] = null;
  }
}

//  UPDATE DISPLAY
function updateStatsDisplay(stats, prefix) {
  ['mp', 'w', 'd', 'l', 'gf', 'ga', 'gd'].forEach(k => {
    const el = document.getElementById(`${prefix}_${k}`);
    if (el) el.textContent = stats[k] ?? '-';
  });
  document.getElementById(`${prefix}_last_5`).textContent = stats.last_5 ?? '-';
  document.getElementById(`${prefix}_top_scorer`).textContent = stats.top_scorer ?? '-';
  document.getElementById(`${prefix}_goalkeeper`).textContent = stats.goalkeeper ?? '-';
}
function clearStatsDisplay(prefix) {
  ['mp','w','d','l','gf','ga','gd','last_5','top_scorer','goalkeeper']
    .forEach(k => {
      const el = document.getElementById(`${prefix}_${k}`);
      if (el) el.textContent = '-';
    });
}

//  PREDICT 
async function compareTeams() {
  try {
    const homeTeam = homeTeamSelect.value;
    const awayTeam = awayTeamSelect.value;

    if (!homeTeam || !awayTeam) {
      predictionResult.innerHTML = '<p>Select both teams first</p>';
      return;
    }
    if (homeTeam === awayTeam) {
      predictionResult.innerHTML = '<p>Pick two different teams.</p>';
      return;
    }

    predictBtn.disabled = true;
    predictBtn.textContent = 'Predicting...';

    const res = await fetch(`${API_BASE_URL}/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam })
    });

    if (!res.ok) {
      const body = await safeJson(res);
      throw new Error(body?.detail || `HTTP ${res.status}`);
    }

    const result = await res.json();
    const homeProb = (result.home_win * 100).toFixed(1);
    const drawProb = (result.draw * 100).toFixed(1);
    const awayProb = (result.away_win * 100).toFixed(1);

    predictionResult.innerHTML = `
      <p><strong>Model:</strong> ${result.details.model}</p>
      <h3>${result.prediction}</h3>
      <p class="home-win">Home Win: ${homeProb}%</p>
      <p class="draw">Draw: ${drawProb}%</p>
      <p class="away-win">Away Win: ${awayProb}%</p>
      <hr>
      <div class="prediction-details">
        <h4>Home Team Form</h4>
        <p>Win Rate: ${result.details.home_team.form.win_ratio}%</p>
        <p>Goals/Game: ${result.details.home_team.form.goals_per_game}</p>
        <p>Defense Rating: ${result.details.home_team.form.defense_rating}</p>

        <h4>Away Team Form</h4>
        <p>Win Rate: ${result.details.away_team.form.win_ratio}%</p>
        <p>Goals/Game: ${result.details.away_team.form.goals_per_game}</p>
        <p>Defense Rating: ${result.details.away_team.form.defense_rating}</p>
      </div>
    `;
  } catch (err) {
    console.error('Prediction error:', err);
    predictionResult.innerHTML = `<p>Error: ${err.message}</p>`;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = 'Predict';
  }
}

//  ENABLE/DISABLE BUTTON 
function updatePredictBtnState() {
  const home = homeTeamSelect.value;
  const away = awayTeamSelect.value;
  predictBtn.disabled = !(home && away && home !== away);
}

//  EVENT
homeTeamSelect.addEventListener('change', async (e) => {
  const t = e.target.value;
  if (t) await loadTeamStats(t, 'home'); else clearStatsDisplay('home');
  updatePredictBtnState();
});

awayTeamSelect.addEventListener('change', async (e) => {
  const t = e.target.value;
  if (t) await loadTeamStats(t, 'away'); else clearStatsDisplay('away');
  updatePredictBtnState();
});

predictBtn.addEventListener('click', compareTeams);

//  INIT 
loadTeams();
