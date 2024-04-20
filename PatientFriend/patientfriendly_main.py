from ChatPatient import ChatPatient

cp = ChatPatient()
tranlate_to = 'bengali'

radiology_report = """No acute cardiopulmonary findings. Specifically, no radiographic evidence of active tuberculosis."
Old granulomatous disease. No acute pulmonary disease.
Large right pleural effusion and patchy left lower lobe airspace disease.
1. No acute cardiopulmonary disease
Heart size normal. Lungs clear. No edema or effusions.
1. No acute cardiopulmonary abnormality. 2. No evidence of active or changes from chronic tuberculosis infection.
Chest. Resolving pulmonary interstitial edema and pulmonary venous hypertension.
Patchy right lower lobe infiltrate as well as probable left basilar infiltrate versus atelectasis.
1. No evidence of active disease.
1. No acute pulmonary abnormality."""

patient_friendly_text = cp.get_friendly_text(radiology_report)

print(patient_friendly_text)
print(f"Translating to {tranlate_to}...")
print(cp.translate_text(patient_friendly_text, tranlate_to))