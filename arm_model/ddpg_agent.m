previousRngState = rng(0, "twister");

obsInfo = rlNumericSpec([2 1], ...
    LowerLimit = [-inf -inf]', ...
    UpperLimit = [inf inf]');

actInfo =  rlNumericSpec([3, 1]);

env = rlSimulinkEnv("TwoSegmentArm_muscles", "TwoSegmentArm_muscles/arm_agent", ...
    obsInfo, actInfo); % Last two might not be correct
% env.ResetFcn = @localResetFcn; % TODO

Ts = 0.2;
Tf = 10;

% Critic Network
obsPath = featureInputLayer(obsInfo.Dimension(1), Name="obsIn");
actPath = featureInputLayer(actInfo.Dimension(1), Name="actIn");

deepLayers = [
    concatenationLayer(1, 2,Name="concat")
    fullyConnectedLayer(5)
    leakyReluLayer()
    fullyConnectedLayer(5)
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
    fullyConnectedLayer(5)
    reluLayer()
    fullyConnectedLayer(5)
    reluLayer()
    fullyConnectedLayer(actInfo.Dimension(1))];

rng(0, "twister");
actorNet = dlnetwork(actorNet);
summary(actorNet)

actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo);

agent = rlDDPGAgent(actor, critic);

agent.AgentOptions.SampleTime = Ts;
agent.AgentOptions.DiscountFactor = 1.0;
agent.AgentOptions.MiniBatchSize = 10;
agent.AgentOptions.ExperienceBufferLength = 1e5;

actorOpts = rlOptimizerOptions( ...
    LearnRate=1e-3, ...
    GradientThreshold=1);
criticOpts = rlOptimizerOptions( ...
    LearnRate=1e-3, ...
    GradientThreshold=1);
agent.AgentOptions.ActorOptimizerOptions = actorOpts;
agent.AgentOptions.CriticOptimizerOptions = criticOpts;

agent.AgentOptions.NoiseOptions.StandardDeviation = 0.3;
agent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 1e-4;


% training options
trainOpts = rlTrainingOptions(...
    MaxEpisodes=500, ...
    MaxStepsPerEpisode=ceil(Tf/Ts), ...
    Plots="training-progress", ...
    Verbose=false, ...
    StopTrainingCriteria="EvaluationStatistic", ...
    StopTrainingValue=8000);

% agent evaluator
evl = rlEvaluator(EvaluationFrequency=10,NumEpisodes=5);

rng(0, "twister");

do_training = true;
if do_training
    trainingStats = train(agent, env, trainOpts, Evaluator=evl);
end

% function in = localResetFcn(in)
% end