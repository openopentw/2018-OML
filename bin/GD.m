function weight = GD(y, X)
    [~, m] = size(X);

    %% parameters
    verbose = 1;
    max_iters = 10000;
    max_alpha_iters = 100;
    w_0 = zeros(m, 1);
    yeta = 1e-2;
    C = 1e-1;
    epsilon = 1e-2;

    %% initialize
    w = w_0;

    % save for speed-up
    X_w = X * w;

    % calculate stopping condition
    f_w = 1 / 2 * (w' * w) + C * sum(log(1 + exp(- y .* X_w)));
    grad_f_w = w + C * (X' * ((1 ./ (1 + exp(- y .* X_w)) - 1) .* y));
    stop_cond_2 = (epsilon ^ 2) * (grad_f_w' * grad_f_w);
    disp(stop_cond_2);

    disp("[iter, iter_alpha, (grad_f_w' * grad_f_w), f_w]");
    for iter = 1: max_iters
        %% calculate direction s
        s = - grad_f_w;
        % save for speed-up
        X_s = X * s;

        %% find alpha
        alpha = 1;
        for iter_alpha = 1: max_alpha_iters
            w_alpha_s = w + alpha * s;
            X_w_alpha_s = X_w + alpha * X_s;
            f_w_alpha_s = 1 / 2 * (w_alpha_s' * w_alpha_s) + ...
                C * sum(log(1 + exp(- y .* X_w_alpha_s)));
            if f_w_alpha_s <= f_w + yeta * alpha * (grad_f_w' * s)
                break
            end

            alpha = alpha / 2;
        end

        %% Update weight
        w = w + alpha * s;
        % save for speed-up
        X_w = X_w + alpha * X_s;
        % calculate grad(f(w)) for next iter
        f_w = 1 / 2 * (w' * w) + C * sum(log(1 + exp(- y .* X_w)));
        grad_f_w = w + C * (X' * ((1 ./ (1 + exp(- y .* X_w)) - 1) .* y));

        if verbose
            disp([iter, iter_alpha, (grad_f_w' * grad_f_w), f_w]);
        end

        if grad_f_w' * grad_f_w < stop_cond_2
            break
        end
    end
    weight = w;
end
