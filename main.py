import pandapower as pp
import pandas as pd
# import opf_3ph_power_voltage
import opf_3ph_current_voltage

#Creating network from json file. This script can be modified in order to use other types of data (xlsx, csv, etc.)
#Additionally, the network can be created by directly defining elements, without any other source of data.
#Replace path with the exact path (all directories) to the neccessary files.

net = pp.from_json(r'path\cigre_lv_modified.json')
#Load curves that are input in pp OPF - both power-voltage and current-voltage formulations

p_load_a = pd.read_excel(r'C:\Users\Tomislav\Desktop\Posao\PES GM 2023\p_load_a.xlsx', index_col = 0)
p_load_b = pd.read_excel(r'C:\Users\Tomislav\Desktop\Posao\PES GM 2023\p_load_b.xlsx', index_col = 0)
p_load_c = pd.read_excel(r'C:\Users\Tomislav\Desktop\Posao\PES GM 2023\p_load_c.xlsx', index_col = 0)

# v_a, v_b, v_c = opf_3ph_power_voltage.opf_3ph_power_voltage(net, p_load_a, p_load_b, p_load_c, vm_pu = 1.00)
v_a, v_b, v_c = opf_3ph_current_voltage.opf_3ph_current_voltage(net, p_load_a, p_load_b, p_load_c, vm_pu = 1.0)


