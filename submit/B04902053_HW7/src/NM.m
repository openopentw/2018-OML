function weight = NM(y, X)
    [~, m] = size(X);

    %% parameters
    verbose = 1;
    max_iters = 10000;
    max_cg_iters = 100;
    max_alpha_iters = 100;
    w_0 = zeros(m, 1);
    yeta = 1e-2;
    xi = 1e-1;
    C = 1e-1;
    epsilon = 1e-2;
    % epsilon = 1e-3;

    % initialize
    w = w_0;

    % save for speed-up
    X_w = X * w;

    % calculate stopping condition
    f_w = 1 / 2 * (w' * w) + C * sum(log(1 + exp(- y .* X_w)));
    grad_f_w = w + C * (X' * ((1 ./ (1 + exp(- y .* X_w)) - 1) .* y));
    stop_cond_2 = (epsilon ^ 2) * (grad_f_w' * grad_f_w);

    disp("[iter, iter_alpha, (grad_f_w' * grad_f_w), f_w]");
    for iter = 1: max_iters
        %% solve Newton Linear System to get the direction `s`
        % initialize
        s = zeros(m, 1);
        r = - grad_f_w;
        d = r;
        % save for calculating grad^2(f(w))d
        e_y_w_x = exp(-y .* X_w);
        D = e_y_w_x ./ ((1 + e_y_w_x) .^ 2);
        % obtain s by solving a system by CG
        for iter_cg = 1: max_cg_iters
            if r' * r <= xi * (grad_f_w' * grad_f_w)
                break
            end
            grad_2_f_w_d = d + C .* (X' * (D .* (X * d)));
            alpha = (r' * r) / (d' * grad_2_f_w_d);
            s = s + alpha .* d;
            old_r = r;
            r = old_r - alpha * grad_2_f_w_d;
            beta_ = (r' * r) / (old_r' * old_r);
            d = r + beta_ .* d;
        end
        % save for speed-up
        X_s = X * s;

        %% Find alpha
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
