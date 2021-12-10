
tic

% INPUTS
GarchAlpha = 0.90; % coefficient of lagged vol/variance
GarchBeta = 0.05; % coefficient of lagged residuals
GarchGamma = 0.05; % coefficient for long term vol/variance

MktVol_Inf  = 0.1000; 
MktRtn  = 0.050; 


% Initialize:
MktVol_d_Inf  = MktVol_Inf ;
MktRtn_d  = MktRtn ;


S = 100000; % simulate S times
M = 50; % number of periods to simulate


% Load seed:
load S_20120921
rng(S_20120921);

VOL_MKT  = nan(S,1);

for ss=1:S

    SampMktRtns = nan(M,1);
    disp(num2str(ss))
    
    % Initialize yesterday's variance (mm=0) to (MktVol_d_Inf ^ 2)
    Sigma_prev = MktVol_d_Inf;
    Eps_prev = (randn * Sigma_prev);
    for tt=1:M

        
        Variance_tt_d = (GarchGamma*(MktVol_d_Inf^2)) + ...
            (GarchAlpha*(Sigma_prev^2)) + ...
            (GarchBeta*(Eps_prev^2));
        Sigma_tt_d = sqrt(Variance_tt_d);

        SampMktRtns(tt) = (randn * Sigma_tt_d) + MktRtn_d;
        Eps_prev = SampMktRtns(tt) - MktRtn_d;
        Sigma_prev = Sigma_tt_d;
        
    end

    
    % Collect more sampling stats:
    VOL_MKT(ss)  = std(SampMktRtns);

end

% Collect median and quartile stats:
STATS_TO_PASTE = nan(5,1); 
STATS_TO_PASTE(:,1) = [mean(VOL_MKT),std(VOL_MKT),quantile(VOL_MKT,[0.25,0.5,0.75])]; 


toc

