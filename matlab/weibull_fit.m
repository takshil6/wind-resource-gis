%% Weibull Distribution Fitting for Wind Resource Assessment
% Fits a 2-parameter Weibull distribution to hourly wind speed data
% from a NOAA station and computes derived energy metrics.
%
% Wind power industry standard for site characterization.
% Reference: IEC 61400-12-1
%
% Usage:
%   1. Run Python pipeline through Phase 1 to generate
%      data/raw/station_wind_hourly.parquet
%   2. Export desired station to CSV (see helper at bottom)
%   3. Update STATION_CSV path below and run

clear; clc; close all;

%% Configuration
STATION_CSV = '../data/interim/station_for_weibull.csv';
HUB_HEIGHT_M = 10;          % rooftop turbine
ROTOR_DIAMETER_M = 1.5;     % small distributed
AIR_DENSITY = 1.225;        % kg/m^3 at sea level, 15 C
RATED_POWER_KW = 1.5;
CUT_IN = 2.5; RATED_WS = 11; CUT_OUT = 25;

%% Load data
T = readtable(STATION_CSV);
ws = T.wind_speed_10m;
ws = ws(~isnan(ws) & ws >= 0);

fprintf('Station data loaded: %d hourly observations\n', length(ws));
fprintf('Mean: %.2f m/s | Std: %.2f m/s | Max: %.2f m/s\n', ...
    mean(ws), std(ws), max(ws));

%% Fit Weibull
% Maximum likelihood estimation
phat = wblfit(ws(ws > 0));   % wblfit needs positive values
c = phat(1);                  % scale parameter (m/s)
k = phat(2);                  % shape parameter (dimensionless)

fprintf('\nWeibull parameters:\n');
fprintf('  Scale (c) = %.3f m/s\n', c);
fprintf('  Shape (k) = %.3f\n', k);

%% Derived metrics
% Mean wind speed from Weibull: c * gamma(1 + 1/k)
weibull_mean = c * gamma(1 + 1/k);
% Wind power density: 0.5 * rho * c^3 * gamma(1 + 3/k)
wpd = 0.5 * AIR_DENSITY * c^3 * gamma(1 + 3/k);
% Rotor swept area
A = pi * (ROTOR_DIAMETER_M / 2)^2;

fprintf('\nDerived metrics:\n');
fprintf('  Empirical mean:    %.3f m/s\n', mean(ws));
fprintf('  Weibull mean:      %.3f m/s\n', weibull_mean);
fprintf('  Wind power density: %.1f W/m^2\n', wpd);
fprintf('  Rotor swept area:   %.2f m^2\n', A);

%% Annual Energy Production (AEP) estimate
% Simple bin method: integrate power curve over Weibull PDF
ws_bins = 0:0.5:30;
bin_centers = ws_bins(1:end-1) + 0.25;
pdf_vals = wblpdf(bin_centers, c, k);
hours_per_bin = pdf_vals * 8760 * 0.5;  % 8760 hr/yr * bin width

power_curve = zeros(size(bin_centers));
for i = 1:length(bin_centers)
    v = bin_centers(i);
    if v < CUT_IN || v >= CUT_OUT
        power_curve(i) = 0;
    elseif v < RATED_WS
        % Cubic ramp from cut-in to rated
        power_curve(i) = RATED_POWER_KW * ((v - CUT_IN) / (RATED_WS - CUT_IN))^3;
    else
        power_curve(i) = RATED_POWER_KW;
    end
end

aep_kwh = sum(power_curve .* hours_per_bin);
capacity_factor = aep_kwh / (RATED_POWER_KW * 8760);

fprintf('\nEnergy production estimate:\n');
fprintf('  AEP:              %.0f kWh/year\n', aep_kwh);
fprintf('  Capacity factor:  %.1f%%\n', capacity_factor * 100);

%% Plots
figure('Position', [100 100 1200 400]);

subplot(1, 3, 1);
histogram(ws, 'Normalization', 'pdf', 'EdgeColor', 'k', 'FaceAlpha', 0.6);
hold on;
x = linspace(0, max(ws), 200);
plot(x, wblpdf(x, c, k), 'r-', 'LineWidth', 2);
xlabel('Wind speed (m/s)'); ylabel('Probability density');
title(sprintf('Weibull fit: c=%.2f, k=%.2f', c, k));
legend('Observed', 'Weibull fit', 'Location', 'best');
grid on;

subplot(1, 3, 2);
plot(bin_centers, power_curve, 'b-', 'LineWidth', 2);
xlabel('Wind speed (m/s)'); ylabel('Power output (kW)');
title('Turbine power curve');
xlim([0 30]); grid on;

subplot(1, 3, 3);
yyaxis left;
bar(bin_centers, hours_per_bin, 'FaceAlpha', 0.5);
ylabel('Hours per year');
yyaxis right;
plot(bin_centers, power_curve .* hours_per_bin, 'r-', 'LineWidth', 2);
ylabel('Energy contribution (kWh)');
xlabel('Wind speed (m/s)');
title('Energy yield by wind speed bin');
xlim([0 30]); grid on;

saveas(gcf, '../outputs/weibull_analysis.png');
fprintf('\nPlot saved to outputs/weibull_analysis.png\n');

%% Save results
results = struct();
results.station_csv = STATION_CSV;
results.weibull_c = c;
results.weibull_k = k;
results.weibull_mean = weibull_mean;
results.wpd = wpd;
results.aep_kwh = aep_kwh;
results.capacity_factor = capacity_factor;
results.n_hours = length(ws);

save('../data/interim/weibull_results.mat', 'results');

% Also export to CSV for Python pipeline to ingest
res_table = struct2table(results);
writetable(res_table, '../data/interim/weibull_results.csv');
fprintf('Results saved to data/interim/weibull_results.{mat,csv}\n');