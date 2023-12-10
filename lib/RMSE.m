function rmse = RMSE(dense_tensor, est_tensor, Omega)
    rmse = sqrt((1/length(nonzeros(~Omega)))*norm(dense_tensor(~Omega)-est_tensor(~Omega),2)^2);
end