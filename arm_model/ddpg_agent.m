function in = localResetFcn(in)
rand = min(max(randn / 2, -1.8), 1.8);
in = setVariable(in, "random_offset", rand);
end

previousRngState = rng(0, "twister");
random_offset = 0;

obsInfo = rlNumericSpec([6 1], ...
    LowerLimit = [-inf -inf -inf -inf -inf -inf]', ...
    UpperLimit = [ inf  inf  inf  inf  inf  inf]');

actInfo =  rlNumericSpec([4, 1]);

env = rlSimulinkEnv("TwoSegmentArm_muscles", "TwoSegmentArm_muscles/arm_agent", ...
    obsInfo, actInfo); % Last two might not be correct
env.ResetFcn = @localResetFcn; % TODO

Ts = 0.075;
Tf = 10;

% Critic Network
obsPath = featureInputLayer(obsInfo.Dimension(1), Name="obsIn");
actPath = featureInputLayer(actInfo.Dimension(1), Name="actIn");

deepLayers = [
    concatenationLayer(1, 2,Name="concat")
    fullyConnectedLayer(15)
    leakyReluLayer()
    dropoutLayer
    fullyConnectedLayer(15)
    leakyReluLayer()
    fullyConnectedLayer(15)
    leakyReluLayer()
    dropoutLayer
    fullyConnectedLayer(15)
    leakyReluLayer()
    fullyConnectedLayer(1, Name="QValue")
    ];

% criticNet = layerGraph();
criticNet = dlnetwork();
criticNet = addLayers(criticNet, obsPath);
criticNet = addLayers(criticNet, actPath);
criticNet = addLayers(criticNet, deepLayers);

criticNet = connectLayers(criticNet, "obsIn", "concat/in1");
criticNet = connectLayers(criticNet, "actIn", "concat/in2");
% plot(criticNet)

rng(0, "twister");
criticNet = initialize(criticNet);
summary(criticNet)

critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
    ObservationInputNames="obsIn", ActionInputNames="actIn");

actorNet = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(10)
    leakyReluLayer()
    dropoutLayer
    fullyConnectedLayer(10)
    leakyReluLayer()
    fullyConnectedLayer(10)
    leakyReluLayer()
    dropoutLayer
    fullyConnectedLayer(10)
    leakyReluLayer()
    fullyConnectedLayer(actInfo.Dimension(1))];

rng(0, "twister");
actorNet = dlnetwork(actorNet);
summary(actorNet)

actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo);

agent = rlDDPGAgent(actor, critic);

agent.AgentOptions.SampleTime = Ts;
agent.AgentOptions.DiscountFactor = 0.9;
agent.AgentOptions.MiniBatchSize = 30;
agent.AgentOptions.ExperienceBufferLength = 1e5;

actorOpts = rlOptimizerOptions( ...
    LearnRate=1e-3, ...
    GradientThreshold=1);
criticOpts = rlOptimizerOptions( ...
    LearnRate=1e-2, ...
    GradientThreshold=1);
agent.AgentOptions.ActorOptimizerOptions = actorOpts;
agent.AgentOptions.CriticOptimizerOptions = criticOpts;

agent.AgentOptions.NoiseOptions.StandardDeviation = 0.3;
agent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 1e-4;


% training options
trainOpts = rlTrainingOptions(...
    MaxEpisodes=3000, ...
    MaxStepsPerEpisode=ceil(Tf/Ts), ...
    Plots="training-progress", ...
    Verbose=false, ...
    StopTrainingCriteria="EvaluationStatistic", ...
    StopTrainingValue=1500);
% UseParallel=true, ...
% trainOpts.ParallelizationOptions.Mode = 'async';
% trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;
% trainOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';

% agent evaluator
evl = rlEvaluator(EvaluationFrequency=50,NumEpisodes=5);

% rng(0, "twister");


