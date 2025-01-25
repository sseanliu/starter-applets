// Copyright 2024 Google LLC

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     https://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

export const colors = [
  "rgb(0, 0, 0)",
  "rgb(255, 255, 255)",
  "rgb(213, 40, 40)",
  "rgb(250, 123, 23)",
  "rgb(240, 186, 17)",
  "rgb(8, 161, 72)",
  "rgb(26, 115, 232)",
  "rgb(161, 66, 244)",
];

export const modelOptions = [
  "models/gemini-2.0-flash-exp",
  "models/gemini-1.5-flash",
];

export const imageOptions: string[] = [
  "origami.jpg",
  "kitchen1.png",
  "clock.jpg",
  "socks.jpg",
  "breakfast.jpg",
  "cat.jpg",
  "spill.jpg",
  "fruit.jpg",
  "baklava.jpg",
];

export const lineOptions = {
  size: 8,
  thinning: 0,
  smoothing: 0,
  streamline: 0,
  simulatePressure: false,
};

export const defaultPromptParts = {
  "2D bounding boxes": [
    "Show me the positions of",
    "SKU items",
    "as a JSON list. Do not return masks. Limit to 25 items.",
  ],
  "3D bounding boxes": [
    "Detect the 3D bounding boxes of",
    "SKU items",
    ', output no more than 10 items. Output a json list where each entry contains the object name in "label" and its 3D bounding box in "box_3d".',
  ],
  Points: [
    "Point to the",
    "SKU items",
    ' with no more than 10 items. The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000.'
  ],
};

export const defaultPrompts = {
  "2D bounding boxes": defaultPromptParts["2D bounding boxes"].join(" "),
  "3D bounding boxes": defaultPromptParts["3D bounding boxes"].join(" "),
  Points: defaultPromptParts.Points.join(" "),
};

const safetyLevel = "only_high";

export const safetySettings = new Map();

safetySettings.set("harassment", safetyLevel);
safetySettings.set("hate_speech", safetyLevel);
safetySettings.set("sexually_explicit", safetyLevel);
safetySettings.set("dangerous_content", safetyLevel);
safetySettings.set("civic_integrity", safetyLevel);
