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

import { useAtom } from "jotai";
import getStroke from "perfect-freehand";
import { useState, useRef, useMemo, useCallback, useEffect } from "react";
import {
  ImageSrcAtom,
  BoundingBoxes2DAtom,
  BoundingBoxes3DAtom,
  ShareStream,
  DetectTypeAtom,
  FOVAtom,
  ImageSentAtom,
  PointsAtom,
  RevealOnHoverModeAtom,
  DrawModeAtom,
  LinesAtom,
  ActiveColorAtom,
  VideoRefAtom,
  ModelSelectedAtom,
} from "./atoms";
import { getSvgPathFromStroke } from "./utils";
import { lineOptions } from "./consts";
import { ResizePayload, useResizeDetector } from "react-resize-detector";
import {GoogleGenerativeAI} from "@google/generative-ai";

const client = new GoogleGenerativeAI(import.meta.env.VITE_GEMINI_API_KEY);

export function Content() {
  const [imageSrc] = useAtom(ImageSrcAtom);
  const [boundingBoxes2D] = useAtom(BoundingBoxes2DAtom);
  const [boundingBoxes3D] = useAtom(BoundingBoxes3DAtom);
  const [stream] = useAtom(ShareStream);
  const [detectType] = useAtom(DetectTypeAtom);
  const [videoRef] = useAtom(VideoRefAtom);
  const [fov] = useAtom(FOVAtom);
  const [, setImageSent] = useAtom(ImageSentAtom);
  const [points] = useAtom(PointsAtom);
  const [revealOnHover] = useAtom(RevealOnHoverModeAtom);
  const [hoverEntered, setHoverEntered] = useState(false);
  const [hoveredBox, _setHoveredBox] = useState<number | null>(null);
  const [drawMode] = useAtom(DrawModeAtom);
  const [lines, setLines] = useAtom(LinesAtom);
  const [activeColor] = useAtom(ActiveColorAtom);
  const [isWebcam, setIsWebcam] = useState(false);
  const [showBboxes, setShowBboxes] = useState(true);
  const [showPoints, setShowPoints] = useState(true);
  const [showItemList, setShowItemList] = useState(false);
  const [lastResponse, setLastResponse] = useState<any>(null);
  const [activeItems, setActiveItems] = useState<Set<string>>(new Set());
  const [relatedItems, setRelatedItems] = useState<{[key: string]: string[]}>({});
  const [modelSelected] = useAtom(ModelSelectedAtom);

  // Handling resize and aspect ratios
  const boundingBoxContainerRef = useRef<HTMLDivElement | null>(null);
  const [containerDims, setContainerDims] = useState({
    width: 0,
    height: 0,
  });
  const [activeMediaDimensions, setActiveMediaDimensions] = useState({
    width: 1,
    height: 1,
  });

  const onResize = useCallback(
    (el: ResizePayload) => {
      if (el.width && el.height) {
        setContainerDims({
          width: el.width,
          height: el.height,
        });
      }
    },
    [],
  );

  const { ref: containerRef } = useResizeDetector({ onResize });

  const boundingBoxContainer = useMemo(() => {
    const { width, height } = activeMediaDimensions;
    const aspectRatio = width / height;
    const containerAspectRatio = containerDims.width / containerDims.height;
    if (aspectRatio < containerAspectRatio) {
      return {
        height: containerDims.height,
        width: containerDims.height * aspectRatio,
      };
    } else {
      return {
        width: containerDims.width,
        height: containerDims.width / aspectRatio,
      };
    }
  }, [containerDims, activeMediaDimensions]);

  // Helper functions
  function matrixMultiply(m: number[][], v: number[]): number[] {
    return m.map((row: number[]) =>
      row.reduce((sum, val, i) => sum + val * v[i], 0),
    );
  }

  const linesAndLabels3D = useMemo(() => {
    if (!boundingBoxContainer) {
      return null;
    }
    let allLines = [];
    let allLabels = [];
    for (const box of boundingBoxes3D) {
      const { center, size, rpy } = box;

      // Convert Euler angles to quaternion
      const [sr, sp, sy] = rpy.map((x) => Math.sin(x / 2));
      const [cr, cp, cz] = rpy.map((x) => Math.cos(x / 2));
      const quaternion = [
        sr * cp * cz - cr * sp * sy,
        cr * sp * cz + sr * cp * sy,
        cr * cp * sy - sr * sp * cz,
        cr * cp * cz + sr * sp * sy,
      ];

      // Calculate camera parameters
      const height = boundingBoxContainer.height;
      const width = boundingBoxContainer.width;
      const f = width / (2 * Math.tan(((fov / 2) * Math.PI) / 180));
      const cx = width / 2;
      const cy = height / 2;
      const intrinsics = [
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
      ];

      // Get box vertices
      const halfSize = size.map((s) => s / 2);
      let corners = [];
      for (let x of [-halfSize[0], halfSize[0]]) {
        for (let y of [-halfSize[1], halfSize[1]]) {
          for (let z of [-halfSize[2], halfSize[2]]) {
            corners.push([x, y, z]);
          }
        }
      }
      corners = [
        corners[1],
        corners[3],
        corners[7],
        corners[5],
        corners[0],
        corners[2],
        corners[6],
        corners[4],
      ];

      // Apply rotation from quaternion
      const q = quaternion;
      const rotationMatrix = [
        [
          1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
          2 * q[0] * q[1] - 2 * q[3] * q[2],
          2 * q[0] * q[2] + 2 * q[3] * q[1],
        ],
        [
          2 * q[0] * q[1] + 2 * q[3] * q[2],
          1 - 2 * q[0] ** 2 - 2 * q[2] ** 2,
          2 * q[1] * q[2] - 2 * q[3] * q[0],
        ],
        [
          2 * q[0] * q[2] - 2 * q[3] * q[1],
          2 * q[1] * q[2] + 2 * q[3] * q[0],
          1 - 2 * q[0] ** 2 - 2 * q[1] ** 2,
        ],
      ];

      const boxVertices = corners.map((corner) => {
        const rotated = matrixMultiply(rotationMatrix, corner);
        return rotated.map((val, idx) => val + center[idx]);
      });

      // Project 3D points to 2D
      const tiltAngle = 90.0;
      const viewRotationMatrix = [
        [1, 0, 0],
        [
          0,
          Math.cos((tiltAngle * Math.PI) / 180),
          -Math.sin((tiltAngle * Math.PI) / 180),
        ],
        [
          0,
          Math.sin((tiltAngle * Math.PI) / 180),
          Math.cos((tiltAngle * Math.PI) / 180),
        ],
      ];

      const points = boxVertices;
      const rotatedPoints = points.map((p) =>
        matrixMultiply(viewRotationMatrix, p),
      );
      const translatedPoints = rotatedPoints.map((p) => p.map((v) => v + 0));
      const projectedPoints = translatedPoints.map((p) =>
        matrixMultiply(intrinsics, p),
      );
      const vertices = projectedPoints.map((p) => [p[0] / p[2], p[1] / p[2]]);

      const topVertices = vertices.slice(0, 4);
      const bottomVertices = vertices.slice(4, 8);

      for (let i = 0; i < 4; i++) {
        const lines = [
          [topVertices[i], topVertices[(i + 1) % 4]],
          [bottomVertices[i], bottomVertices[(i + 1) % 4]],
          [topVertices[i], bottomVertices[i]],
        ];

        for (let [start, end] of lines) {
          const dx = end[0] - start[0];
          const dy = end[1] - start[1];
          const length = Math.sqrt(dx * dx + dy * dy);
          const angle = Math.atan2(dy, dx);

          allLines.push({ start, end, length, angle });
        }
      }

      // Add label with fade effect
      const textPosition3d = points[0].map(
        (_, idx) => points.reduce((sum, p) => sum + p[idx], 0) / points.length,
      );
      textPosition3d[2] += 0.1;

      const textPoint = matrixMultiply(
        intrinsics,
        matrixMultiply(
          viewRotationMatrix,
          textPosition3d.map((v) => v + 0),
        ),
      );
      const textPos = [
        textPoint[0] / textPoint[2],
        textPoint[1] / textPoint[2],
      ];
      allLabels.push({ label: box.label, pos: textPos });
    }
    return [allLines, allLabels] as const;
  }, [boundingBoxes3D, boundingBoxContainer, fov]);

  function setHoveredBox(e: React.PointerEvent) {
    const boxes = document.querySelectorAll(".bbox");
    const dimensionsAndIndex = Array.from(boxes).map((box, i) => {
      const { top, left, width, height } = box.getBoundingClientRect();
      return {
        top,
        left,
        width,
        height,
        index: i,
      };
    });
    // Sort smallest to largest
    const sorted = dimensionsAndIndex.sort(
      (a, b) => a.width * a.height - b.width * b.height,
    );
    // Find the smallest box that contains the mouse
    const { clientX, clientY } = e;
    const found = sorted.find(({ top, left, width, height }) => {
      return (
        clientX > left &&
        clientX < left + width &&
        clientY > top &&
        clientY < top + height
      );
    });
    if (found) {
      _setHoveredBox(found.index);
    } else {
      _setHoveredBox(null);
    }
  }

  const downRef = useRef<Boolean>(false);

  useEffect(() => {
    if (stream) {
      setIsWebcam(stream.getVideoTracks()[0].kind === 'video');
    } else {
      setIsWebcam(false);
    }
  }, [stream]);

  // Subscribe to response updates from Prompt component
  useEffect(() => {
    const handleResponse = (event: CustomEvent) => {
      setLastResponse(event.detail);
    };
    window.addEventListener('parsedResponse', handleResponse as EventListener);
    return () => {
      window.removeEventListener('parsedResponse', handleResponse as EventListener);
    };
  }, []);

  // Get unique item names from the response
  const getItemList = useMemo(() => {
    if (!lastResponse) return [] as string[];
    return Array.from(new Set(lastResponse.map((item: any) => item.label))).sort() as string[];
  }, [lastResponse]);

  // Function to analyze related objects
  const analyzeRelatedObjects = async (item: string) => {
    try {
      const result = await client
        .getGenerativeModel(
          {model: modelSelected},
          {apiVersion: 'v1beta'}
        )
        .generateContent({
          contents: [{
            role: "user",
            parts: [{
              text: `Given this ${item} in the current scene, list the most related objects that are present in the scene from this list: ${getItemList.join(", ")}. Output a JSON array where each entry is just the object name as a string. Consider spatial relationships, functional relationships, and common usage patterns. Example format: ["object1", "object2", "object3"]. Do not include explanations, just the JSON array.`
            }]
          }]
        });
      
      const text = await result.response.text();
      console.log('Related items response:', text);
      
      // Parse JSON response
      let relatedObjects: string[] = [];
      try {
        // Extract JSON if it's wrapped in markdown code blocks
        const jsonText = text.includes('```') ? 
          text.split('```json')[1]?.split('```')[0] || text :
          text;
        relatedObjects = JSON.parse(jsonText)
          .filter((item: string) => getItemList.includes(item));
      } catch (e) {
        console.error('Error parsing related items JSON:', e);
        // Fallback to old text parsing if JSON parse fails
        relatedObjects = text
          .split(/[,\n]/)
          .map(item => item.trim())
          .filter(item => getItemList.includes(item));
      }
      
      console.log('Filtered related objects:', relatedObjects);
      
      setRelatedItems(prev => ({
        ...prev,
        [item]: relatedObjects
      }));
    } catch (error) {
      console.error('Error analyzing related objects:', error);
    }
  };

  return (
    <div ref={containerRef} className="w-full grow relative">
      {/* Always visible item list */}
      <div className="absolute left-4 top-4 z-10 bg-white border border-gray-200 rounded-lg shadow-lg p-4 min-w-[200px]">
        <div className="font-medium mb-2">Detected Items ({getItemList.length}):</div>
        <ul className="list-disc pl-5">
          {getItemList.map((item) => {
            const isActive = activeItems.has(item);
            const isRelated = Object.entries(relatedItems).some(([activeItem, related]) => 
              activeItems.has(activeItem) && related.includes(item)
            );
            
            return (
              <li key={item}>
                <div 
                  className={`text-sm cursor-pointer transition-colors ${
                    isActive ? "text-[#3B68FF] font-medium" : 
                    isRelated ? "text-[#22c55e] font-medium" : ""
                  } hover:text-[#3B68FF]`}
                  onClick={() => {
                    setActiveItems(prev => {
                      const newSet = new Set(prev);
                      if (newSet.has(item)) {
                        newSet.delete(item);
                        // Clear related items when deactivated
                        setRelatedItems(prev => {
                          const newRelated = { ...prev };
                          delete newRelated[item];
                          return newRelated;
                        });
                      } else {
                        newSet.add(item);
                        // Analyze related objects when activated
                        analyzeRelatedObjects(item);
                      }
                      return newSet;
                    });
                  }}
                >
                  {item}
                  {relatedItems[item] && relatedItems[item].length > 0 && (
                    <div className="ml-4 mt-1 text-xs text-gray-600">
                      Related: {relatedItems[item].join(", ")}
                    </div>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      {/* Add toggle buttons when in 2D mode */}
      {detectType === "2D bounding boxes" && boundingBoxes2D.length > 0 && (
        <div className="absolute top-4 right-4 flex gap-2 z-10">
          <button
            className={`px-3 py-1 rounded text-sm ${showBboxes ? "bg-[#3B68FF] text-white" : "bg-gray-200 text-gray-700"}`}
            onClick={() => setShowBboxes(!showBboxes)}
          >
            Boxes
          </button>
          <button
            className={`px-3 py-1 rounded text-sm ${showPoints ? "bg-[#3B68FF] text-white" : "bg-gray-200 text-gray-700"}`}
            onClick={() => setShowPoints(!showPoints)}
          >
            Points
          </button>
        </div>
      )}
      {stream ? (
        <video
          className="absolute top-0 left-0 w-full h-full object-contain"
          autoPlay
          playsInline
          muted={isWebcam}
          onLoadedMetadata={(e) => {
            setActiveMediaDimensions({
              width: e.currentTarget.videoWidth,
              height: e.currentTarget.videoHeight,
            });
          }}
          ref={(video) => {
            videoRef.current = video;
            if (video && !video.srcObject) {
              video.srcObject = stream;
            }
          }}
        />
      ) : imageSrc ? (
        <img
          src={imageSrc}
          className="absolute top-0 left-0 w-full h-full object-contain"
          alt="Uploaded image"
          onLoad={(e) => {
            setActiveMediaDimensions({
              width: e.currentTarget.naturalWidth,
              height: e.currentTarget.naturalHeight,
            });
          }}
        />
      ) : null}
      <div
        className={`absolute w-full h-full left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 ${hoverEntered ? "hide-box" : ""} ${drawMode ? "cursor-crosshair" : ""}`}
        ref={boundingBoxContainerRef}
        onPointerEnter={(e) => {
          if (revealOnHover && !drawMode) {
            setHoverEntered(true);
            setHoveredBox(e);
          }
        }}
        onPointerMove={(e) => {
          if (revealOnHover && !drawMode) {
            setHoverEntered(true);
            setHoveredBox(e);
          }
          if (downRef.current) {
            const parentBounds =
              boundingBoxContainerRef.current!.getBoundingClientRect();
            setLines((prev) => [
              ...prev.slice(0, prev.length - 1),
              [
                [
                  ...prev[prev.length - 1][0],
                  [
                    (e.clientX - parentBounds.left) /
                    boundingBoxContainer!.width,
                    (e.clientY - parentBounds.top) /
                    boundingBoxContainer!.height,
                  ],
                ],
                prev[prev.length - 1][1],
              ],
            ]);
          }
        }}
        onPointerLeave={(e) => {
          if (revealOnHover && !drawMode) {
            setHoverEntered(false);
            setHoveredBox(e);
          }
        }}
        onPointerDown={(e) => {
          if (drawMode) {
            setImageSent(false);
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
            downRef.current = true;
            const parentBounds =
              boundingBoxContainerRef.current!.getBoundingClientRect();
            setLines((prev) => [
              ...prev,
              [
                [
                  [
                    (e.clientX - parentBounds.left) /
                    boundingBoxContainer!.width,
                    (e.clientY - parentBounds.top) /
                    boundingBoxContainer!.height,
                  ],
                ],
                activeColor,
              ],
            ]);
          }
        }}
        onPointerUp={(e) => {
          if (drawMode) {
            (e.target as HTMLElement).releasePointerCapture(e.pointerId);
            downRef.current = false;
          }
        }}
        style={{
          width: boundingBoxContainer.width,
          height: boundingBoxContainer.height,
        }}
      >
        {lines.length > 0 && (
          <svg
            className="absolute top-0 left-0 w-full h-full"
            style={{
              pointerEvents: "none",
              width: boundingBoxContainer?.width,
              height: boundingBoxContainer?.height,
            }}
          >
            {lines.map(([points, color], i) => (
              <path
                key={i}
                d={getSvgPathFromStroke(
                  getStroke(
                    points.map(([x, y]) => [
                      x * boundingBoxContainer!.width,
                      y * boundingBoxContainer!.height,
                      0.5,
                    ]),
                    lineOptions,
                  ),
                )}
                fill={color}
              />
            ))}
          </svg>
        )}
        {detectType === "2D bounding boxes" &&
          boundingBoxes2D.map((box, i) => {
            const isActive = activeItems.has(box.label);
            const isRelated = Object.entries(relatedItems).some(([activeItem, related]) => 
              activeItems.has(activeItem) && related.includes(box.label)
            );
            
            return (
              <div key={i}>
                {showBboxes && (
                  <div
                    className={`absolute bbox border-2 ${
                      isActive ? "border-[#ff3b3b]" : 
                      isRelated ? "border-[#22c55e]" :
                      "border-[#3B68FF]"
                    } ${i === hoveredBox ? "reveal" : ""}`}
                    style={{
                      transformOrigin: "0 0",
                      top: box.y * 100 + "%",
                      left: box.x * 100 + "%",
                      width: box.width * 100 + "%",
                      height: box.height * 100 + "%",
                    }}
                  >
                    <div className={`${
                      isActive ? "bg-[#ff3b3b]" : 
                      isRelated ? "bg-[#22c55e]" :
                      "bg-[#3B68FF]"
                    } text-white absolute left-0 top-0 text-sm px-1`}>
                      {box.label}
                    </div>
                  </div>
                )}
                {showPoints && (
                  <div
                    className="absolute bg-red"
                    style={{
                      left: `${(box.x + box.width/2) * 100}%`,
                      top: `${(box.y + box.height/2) * 100}%`,
                    }}
                  >
                    <div className={`absolute ${
                      isActive ? "bg-[#ff3b3b]" : 
                      isRelated ? "bg-[#22c55e]" :
                      "bg-[#3B68FF]"
                    } text-center text-white text-xs px-1 bottom-4 rounded-sm -translate-x-1/2 left-1/2`}>
                      {box.label}
                    </div>
                    <div className={`absolute w-4 h-4 ${
                      isActive ? "bg-[#ff3b3b]" : 
                      isRelated ? "bg-[#22c55e]" :
                      "bg-[#3B68FF]"
                    } rounded-full border-white border-[2px] -translate-x-1/2 -translate-y-1/2`}></div>
                  </div>
                )}
              </div>
            );
          })}
        {detectType === "Points" &&
          points.map((point, i) => {
            return (
              <div
                key={i}
                className="absolute bg-red"
                style={{
                  left: `${point.point.x * 100}%`,
                  top: `${point.point.y * 100}%`,
                }}
              >
                <div className="absolute bg-[#3B68FF] text-center text-white text-xs px-1 bottom-4 rounded-sm -translate-x-1/2 left-1/2">
                  {point.label}
                </div>
                <div className="absolute w-4 h-4 bg-[#3B68FF] rounded-full border-white border-[2px] -translate-x-1/2 -translate-y-1/2"></div>
              </div>
            );
          })}
        {detectType === "3D bounding boxes" && linesAndLabels3D ? (
          <>
            {linesAndLabels3D[0].map((line, i) => (
              <div
                key={i}
                className="absolute h-[2px] bg-[#3B68FF]"
                style={{
                  width: `${line.length}px`,
                  transform: `translate(${line.start[0]}px, ${line.start[1]}px) rotate(${line.angle}rad)`,
                  transformOrigin: "0 0",
                }}
              ></div>
            ))}
            {linesAndLabels3D[1].map((label, i) => (
              <div
                key={i}
                className="absolute bg-[#3B68FF] text-white text-xs px-1"
                style={{
                  top: `${label.pos[1]}px`,
                  left: `${label.pos[0]}px`,
                  transform: "translate(-50%, -50%)",
                }}
              >
                {label.label}
              </div>
            ))}
          </>
        ) : null}
      </div>
    </div>
  );
}
