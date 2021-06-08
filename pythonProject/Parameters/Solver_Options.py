iter_step = 1
opts_IPOPT = {}
opts_IPOPT["calc_f"] = True
opts_IPOPT["calc_g"] = True
opts_IPOPT["iteration_callback_step"] = iter_step


opts_SQP = {}
opts_SQP["calc_f"] = True
opts_SQP["calc_g"] = True
opts_SQP["iteration_callback_step"] = iter_step