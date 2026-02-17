# Hybrid-AAN-and-RAN-control
This repository contains the open source code linked to the paper: "Myoelectric and Hybrid AAN - RAN control of exoskeleton arm", introducing the two hybrid AAN/RAN controllers Performance-Adaptive Tracking-Error based Resist and Assist As Needed hybrid control (PATE-RAAN) and Directional Intent-Aware hybrid Resist and Assist As Needed hybrid control (DIA-RAAN).

# Contents
This repository contains the following files:
- ControlMode.py            -> contains a class used to define the two control strategies.
- HysteresisThreshold.py    -> contains the hysteresis thresholding operation used.
- ILC.py                    -> Contains the Iterative Learning Controller.
- OIAC.py                   -> Contains the Online Impedance Adaption Controller.
- PATERAN.py                -> Contains the AAN/RAN hybrid PATE-RAAN controller introduced in section 2.5.1 of the paper.
- DIARAAN.py                -> Contains the AAN/RAN hybrid DIA-RAAN controller introduced in section 2.5.2 of the paper.
- Template.py               -> A template file of a control loop, showing the intended use of the controllers, and how to implement them.

# Dependencies
All requirements are in the Requirements.txt file and can be installed using the following command:
 - pip install -r requirements.txt
The only other dependency is the OIAC controller code, which has been imported from the following repository: ??? by ???.
