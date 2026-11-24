// k6 run load_tests/k6_50k_load.js   (BASE_URL, AUTH_SECRET-issued tokens via /v1/dev/token in DEV_MODE)
// Ramp 2m->10K, 3m->50K, hold 5m, down 2m. Thresholds mirror KPI-31..34. A single machine cannot generate 50K VUs:
// run distributed (k6 operator / cloud) against a horizontally scaled deployment.
import http from 'k6/http';
import { check, sleep } from 'k6';

const BASE = __ENV.BASE_URL || 'http://localhost:8000';
const QUERIES = ['What is the refund policy?', 'How many days of PTO do employees get per year?', 'How long are audit logs retained?',
  'What is the restocking fee for opened electronics?', 'How long is the hardware warranty?', 'What is the parental leave policy in Norway?'];

export const options = {
  stages: [{ duration: '2m', target: 10000 }, { duration: '3m', target: 50000 }, { duration: '5m', target: 50000 }, { duration: '2m', target: 0 }],
  thresholds: {
    http_req_duration: ['p(50)<240', 'p(95)<450', 'p(99)<590'],
    http_req_failed: ['rate<0.0005'],
    checks: ['rate>0.999'],
  },
};

export function setup() {
  const tokens = {};
  for (let i = 0; i < 50; i++) {
    const r = http.post(`${BASE}/v1/dev/token`, JSON.stringify({ tenant_id: `tenant_${i}`, user_id: 'user_load', role: 'user' }), { headers: { 'Content-Type': 'application/json' } });
    tokens[i] = r.json('token');
  }
  return tokens;
}

export default function (tokens) {
  const t = __VU % 50;
  const res = http.post(`${BASE}/v1/ask`, JSON.stringify({ query: QUERIES[(__VU + __ITER) % QUERIES.length] }),
    { headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${tokens[t]}` } });
  check(res, {
    'status 200': (r) => r.status === 200,
    'has trace id': (r) => !!r.json('trace_id'),
    'no cross-tenant leak': (r) => !(r.json('answer') || '').includes('12 percent'),   // Acme-only fact must never reach other tenants
  });
  sleep(1);
}
