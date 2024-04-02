In the file main.py, replace "path" with the exact path to your files. 
Also, if necessary, replace the name of Excel files defining active power values (active_a_time, active_b_time, active_c_time). 
The structure of the files needs to be the same as in the example files in order for the code to work properly.

Excel files der_a, der_b, der_c predefine the connection phase of DERs (1 means connected, 0 means disconnected). 
While you need to keep the same file structure, it is possible to define your own way of determining the connection phase.

The test network used in the example is the benchmark CIGRE LV network.
To run calculations on your own network, you need to define network elements following the pandapower structure. You need to define the following elements: external grid, transformers, buses, lines and asymmetric loads.

In the **main.py**, you can choose from three unbalanced OPF formulations.
_Current-voltage_ and _power-voltage_ are nonlinear nonconvex formulations and _lindist_ is linearised formulation.

In Python scripts **opf_3ph_current_voltage.py**, **opf_3ph_power_voltage.py**, and **opf_3ph_lindist.py**, you can change the objective function and other parameters if necessary.
Maximizing DER export is predefined objective function. If you want to maximize import limits, you need to change the objective function by changing _sense = pyo.maximize_ to _sense = pyo.minimize_.
Furthermore, you need to change the range of allowable values of DER (bounds) by modifying _model.p_der = pyo.Var(buses, phases, times, within = pyo.Reals, **bounds = (0,100)**, initialize = 0.0)_ to
_model.p_der = pyo.Var(buses, phases, times, within = pyo.Reals, **bounds = (-100,0)**, initialize = 0.0)_.

With or wihout modifying the initial scripts, you just need to run the **main.py** and as a result, the code will return the objective function value, voltage magnitude, apparent power flow and active DER power.

If you notice that something is wrong with the code or you need help with setting up the tool, you can contact me at **tomislav.antic@fer.hr**
