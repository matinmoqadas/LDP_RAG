"""Run with Python after installing ldp-rag; no providers or downloads needed."""
from decimal import Decimal
from ldp_rag import RandomizedResponse, BoundedNumeric, TextRandomizer, BudgetAccountant

category = RandomizedResponse(('public', 'sensitive', 'confidential'), epsilon=1)
print('Category report:', category.randomize('sensitive'))
print('Exact worst-case likelihood ratio:', category.likelihood_ratio)

number = BoundedNumeric(lower=0, upper=100, bins=11, epsilon=1)
print('Numeric report:', number.randomize(42))

# Use a stable private unit ID and an existing writable directory. Repeated runs
# share this ledger and stop at the limit; do not delete it to bypass the limit.
text = TextRandomizer(('public', 'policy', 'medical', 'service'), max_tokens=4, epsilon=Decimal('.5'))
with BudgetAccountant(Decimal('2'), path='example-privacy.sqlite') as accountant:
    accountant.reserve('example-user', text.epsilon)
    print('Fixed-size text report:', text.randomize_text('medical record for Alice'))
    print('Budget spent:', accountant.spent('example-user'))
