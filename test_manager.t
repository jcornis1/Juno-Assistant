//!------ Test file for mode: manager ------!
//

apikey qwen_4b = "http://localhost:11434";

switchmode manager = "qwen_4b";
modeInstructions["manager"] = "You manage the feature and verify it's complete.";

currentMode = "manager";
callMode("Review the feature implementation and check for gaps.");
