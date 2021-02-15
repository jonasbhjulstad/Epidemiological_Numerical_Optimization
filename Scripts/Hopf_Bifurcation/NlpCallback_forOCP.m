classdef NlpCallback_forOCP < casadi.Callback
  properties
    data
    nx
    ng
  end
  methods
    function self = NlpCallback_forOCP(name, nx, ng)
       self@casadi.Callback();
       self.data = [];
       self.nx = nx;
       self.ng = ng;
      construct(self, name);
    end

    function [returncode] = eval(self, arg)
        self.data = [ self.data full(arg{1})];
        if exist('save_iter.mat')==0
            data=full(arg{1});
        else
            load('save_iter.mat');
            data=[data full(arg{1})];
        end
        save('save_iter.mat','data');
      
    end
  end
end