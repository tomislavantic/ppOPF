import pandas as pd
import pandapower as pp
import opf_3ph_current_voltage
import opf_3ph_power_voltage
import opf_3ph_lindist

p_a = pd.read_excel(r'path\CIGRE\active_a_time.xlsx', index_col = 0)
p_b = pd.read_excel(r'path\CIGRE\active_b_time.xlsx', index_col = 0)
p_c = pd.read_excel(r'path\CIGRE\active_c_time.xlsx', index_col = 0)

der_a = pd.read_excel(r'C:\Users\Tomislav\Desktop\Posao\SEGAN SEST Extension\CIGRE\der_a.xlsx', index_col = 0)
der_b = pd.read_excel(r'C:\Users\Tomislav\Desktop\Posao\SEGAN SEST Extension\CIGRE\der_b.xlsx', index_col = 0)
der_c = pd.read_excel(r'C:\Users\Tomislav\Desktop\Posao\SEGAN SEST Extension\CIGRE\der_c.xlsx', index_col = 0)

net = pp.from_excel(r'C:\Users\Tomislav\Desktop\Posao\SEGAN SEST Extension\CIGRE\cigre_lv.xlsx')

obj, total, v_a, v_b, v_c, p_der_a, p_der_b, p_der_c, s_a, s_b, s_c = opf_3ph_current_voltage.opf_3ph_current_voltage(net, p_a, p_b, p_c, der_a, der_b, der_c, 1.0)
# obj, total, v_a, v_b, v_c, p_der_a, p_der_b, p_der_c, s_a, s_b, s_c = opf_3ph_power_voltage.opf_3ph_power_voltage(net, p_a, p_b, p_c, der_a, der_b, der_c, 1.0)
# obj, total, v_a, v_b, v_c, p_der_a, p_der_b, p_der_c, s_a, s_b, s_c = opf_3ph_lindist.opf_3ph_lindistflow(net, p_a, p_b, p_c, der_a, der_b, der_c, 1.0)

v_a.to_excel(r'path\v_a_s1_cs_1_2.xlsx')
v_b.to_excel(r'path\v_b_s1_cs_1_2.xlsx')
v_c.to_excel(r'path\v_c_s1_cs_1_2.xlsx')

p_der_a.to_excel(r'path\p_der_a_s1_cs_1_2.xlsx')
p_der_b.to_excel(r'path\p_der_b_s1_cs_1_2.xlsx')
p_der_c.to_excel(r'path\p_der_c_s1_cs_1_2.xlsx')

s_a.to_excel(r'path\s_a_s1_cs_1_2.xlsx')
s_b.to_excel(r'path\s_b_s1_cs_1_2.xlsx')
s_c.to_excel(r'path\s_c_s1_cs_1_2.xlsx')
