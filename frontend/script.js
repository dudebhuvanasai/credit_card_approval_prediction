const form = document.getElementById('app-form');
const submitBtn = document.getElementById('submit-btn');
const accuracyBadge = document.getElementById('accuracy-badge');

const resultPanel = document.getElementById('result-panel');
const resultIcon = document.getElementById('result-icon');
const resultTitle = document.getElementById('result-title');
const resultCopy = document.getElementById('result-copy');
const probRow = document.getElementById('prob-row');
const probValue = document.getElementById('prob-value');
const probFill = document.getElementById('prob-fill');

function val(id){ return document.getElementById(id).value; }
function checked(id){ return document.getElementById(id).checked; }

// Load model info for the accuracy badge
fetch('/api/health')
  .then(r => r.json())
  .then(data => {
    const pct = data.accuracy != null ? (data.accuracy * 100).toFixed(2) + '%' : 'n/a';
    accuracyBadge.textContent = `Model hold-out accuracy: ${pct}`;
  })
  .catch(() => {
    accuracyBadge.textContent = 'Model info unavailable';
  });

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = 'Predicting…';

  const yearsEmployed = Number(val('years_employed'));

  const payload = {
    gender: val('gender'),
    own_car: val('own_car'),
    own_realty: val('own_realty'),
    children: Number(val('children')),
    family_members: Number(val('family_members')),
    family_status: val('family_status'),
    annual_income: Number(val('annual_income')),
    income_type: val('income_type'),
    occupation: val('occupation'),
    is_employed: yearsEmployed > 0,
    years_employed: yearsEmployed,
    age: Number(val('age')),
    education: val('education'),
    housing_type: val('housing_type'),
    has_work_phone: checked('has_work_phone'),
    has_phone: checked('has_phone'),
    has_email: checked('has_email'),
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');
    renderResult(data);
  } catch (err) {
    resultPanel.className = 'result-panel';
    resultIcon.textContent = '⚠️';
    resultTitle.textContent = 'Something went wrong';
    resultCopy.textContent = err.message;
    probRow.style.display = 'none';
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Predict Approval';
  }
});

function renderResult(data){
  const approved = data.decision === 'APPROVED';
  const goodCreditProb = 1 - data.risk_probability;
  const pct = (goodCreditProb * 100).toFixed(1);

  resultPanel.className = 'result-panel ' + (approved ? 'approved' : 'declined');
  resultIcon.textContent = approved ? '✅' : '❌';
  resultTitle.textContent = approved ? 'Likely Approved' : 'Likely Rejected';
  resultCopy.textContent = approved
    ? 'Based on the details provided, this application is likely to be approved.'
    : 'Based on the details provided, this application is likely to be rejected.';

  probRow.style.display = 'block';
  probValue.textContent = pct + '%';
  probFill.style.width = pct + '%';

  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
