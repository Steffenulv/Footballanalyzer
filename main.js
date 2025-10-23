// Frontend logic for the Serie A predictor
(function(){
  const teams = [
    'Atalanta','Bologna','Cremonese','Empoli','Fiorentina','Inter','Juventus','Lazio','Milan','Monza','Napoli','Roma','Salernitana','Sassuolo','Spezia','Torino','Udinese','Verona'
  ];

  // Elements
  const homeSelect = document.getElementById('homeTeam');
  const awaySelect = document.getElementById('awayTeam');
  const form = document.getElementById('predict-form');
  const result = document.getElementById('result');
  const explainText = document.getElementById('explainText');
  const backendIndicator = document.getElementById('backend-indicator');

  const homeBar = document.getElementById('homeBar');
  const drawBar = document.getElementById('drawBar');
  const awayBar = document.getElementById('awayBar');
  const homeProb = document.getElementById('homeProb');
  const drawProb = document.getElementById('drawProb');
  const awayProb = document.getElementById('awayProb');

  // Populate teams
  function populateTeams(){
    teams.forEach(t=>{
      const o1 = document.createElement('option'); o1.value=t; o1.textContent=t;
      const o2 = o1.cloneNode(true);
      homeSelect.appendChild(o1);
      awaySelect.appendChild(o2);
    });
    homeSelect.selectedIndex = 0;
    awaySelect.selectedIndex = 1;
  }

  populateTeams();

  // Simple client-side heuristic predictor used as fallback
  // Returns probs object: {home: 0..1, draw: 0..1, away: 0..1} and explanation string
  function heuristicPredict(inputs){
    // inputs: {homeForm, awayForm, homeInjuries, awayInjuries, homeAdv}
    const hf = Number(inputs.homeForm)||5;
    const af = Number(inputs.awayForm)||5;
    const hi = Number(inputs.homeInjuries)||0;
    const ai = Number(inputs.awayInjuries)||0;
    const adv = Number(inputs.homeAdv)||1;

    // Base strength
    let homeScore = hf*1.2 - hi*0.8 + adv*1.5;
    let awayScore = af*1.2 - ai*0.8;

    // Slight league/team effect by name length (deterministic, cheap)
    homeScore += (inputs.homeTeam.length % 3) * 0.2;
    awayScore += (inputs.awayTeam.length % 3) * 0.15;

    // Map to probabilities using softmax-like transform with draw boost when scores close
    const diff = homeScore - awayScore;
    const closeness = Math.exp(-Math.abs(diff)/1.5);

    let ph = Math.max(0, 0.4 + diff*0.05);
    let pa = Math.max(0, 0.3 - diff*0.05);
    let pd = 0.2 + 0.6*closeness;

    // Normalize
    const sum = ph+pd+pa;
    ph/=sum; pd/=sum; pa/=sum;

    const explanation = `Heuristic used. Inputs:\nHome form=${hf}, Away form=${af}, Home injuries=${hi}, Away injuries=${ai}, Home adv=${adv}\nComputed scores: home=${homeScore.toFixed(2)}, away=${awayScore.toFixed(2)}, diff=${diff.toFixed(2)}\nCloseness=${closeness.toFixed(2)}.`;

    return {probs:{home:ph,draw:pd,away:pa},explanation};
  }

  // Render probabilities into UI
  function renderProbs(probs){
    const toPct = n => Math.round(n*100);
    homeBar.style.width = toPct(probs.home)+'%';
    drawBar.style.width = toPct(probs.draw)+'%';
    awayBar.style.width = toPct(probs.away)+'%';
    homeProb.textContent = `${toPct(probs.home)}%`;
    drawProb.textContent = `${toPct(probs.draw)}%`;
    awayProb.textContent = `${toPct(probs.away)}%`;
  }

  // Try backend call; expects POST /api/predict with JSON -> {probs:{home,draw,away},explanation}
  async function callBackend(payload){
    try{
      const res = await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),cache:'no-store'});
      if(!res.ok) throw new Error('non-2xx');
      const data = await res.json();
      if(data && data.probs) return data;
      throw new Error('invalid payload');
    }catch(err){
      throw err;
    }
  }

  // Main submit handler
  form.addEventListener('submit', async (ev)=>{
    ev.preventDefault();
    const payload = Object.fromEntries(new FormData(form).entries());
    payload.homeTeam = homeSelect.value;
    payload.awayTeam = awaySelect.value;

    // Disable button while predicting
    const btn = document.getElementById('predictBtn');
    btn.disabled = true; btn.textContent = 'Predicting...';

    // Try backend first, else fallback to heuristic
    let res;
    try{
      const backend = await callBackend(payload);
      backendIndicator.textContent = 'connected';
      backendIndicator.style.color = '#8ef';
      res = {probs:backend.probs, explanation: backend.explanation || 'Backend result.'};
    }catch(e){
      backendIndicator.textContent = 'not connected';
      backendIndicator.style.color = '';
      res = heuristicPredict(payload);
    }

    renderProbs(res.probs);
    explainText.textContent = res.explanation || 'No explanation provided.';
    btn.disabled = false; btn.textContent = 'Predict';
  });

  // Reset handler
  document.getElementById('resetBtn').addEventListener('click', ()=>{
    form.reset();
    homeSelect.selectedIndex = 0; awaySelect.selectedIndex = 1;
    renderProbs({home:0,draw:0,away:0});
    explainText.textContent = 'No prediction yet.';
  });

  // Accessibility: ensure away != home on change
  function ensureDifferent(){
    if(homeSelect.value === awaySelect.value){
      // pick next available for away
      const idx = (homeSelect.selectedIndex+1) % homeSelect.options.length;
      awaySelect.selectedIndex = idx;
    }
  }
  homeSelect.addEventListener('change', ensureDifferent);
  awaySelect.addEventListener('change', ensureDifferent);

  // Initialize empty result
  renderProbs({home:0,draw:0,away:0});
})();
