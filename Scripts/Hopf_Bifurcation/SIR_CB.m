classdef SIR_CB < casadi.Callback
  properties
    
  end
  methods
    function self = SIR_CB(name)
      self@casadi.Callback();
      construct(self, name);
    end

    % Number of inputs and outputs
    function v=get_n_in(self)
      v=4;
    end
    function v=get_n_out(self)
      v=1;
    end

    function v = get_sparsity_in(self, i)
        if i == 0
            v = casadi.Sparsity.dense(273,1);
        else
            v = casadi.Sparsity.dense(1,1);
        end
    end
    function v = get_sparsity_out(self, i)
        if i == 0
            v = casadi.Sparsity.dense(273,260);
        end
    end
        
    % Initialize the object
    function init(self)
      disp('initializing object')
    end

    % Evaluate numerically
    function w_log = eval(self, arg) 
        z = full(arg{1});
        p1 = full(arg{2});
        p2 = full(arg{3});
        p3 = full(arg{4});
        [c,ceq,dc,dceq] = confun(z,p1,p2,p3,self.para);
        dceq = {dceq};

    end
  end
end