from qiskit_ibm_runtime import Estimator
from qiskit_aer import AerSimulator


def run_estimator(pubs, shots):
    backend = AerSimulator()
    estimator = Estimator(mode=backend, options={'default_shots': shots})

    job = estimator.run(pubs)
    return job.result()
