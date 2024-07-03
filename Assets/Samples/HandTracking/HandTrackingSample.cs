using System.Collections.Generic;
using TensorFlowLite;
using TextureSource;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(VirtualTextureSource))]
public class HandTrackingSample : MonoBehaviour
{
    [Header("Model Settings")]
    [SerializeField, FilePopup("*.tflite")]
    private string palmModelFile = "palm_detection.tflite";

    [SerializeField, FilePopup("*.tflite")]
    private string landmarkModelFile = "hand_landmark.tflite";

    [SerializeField]
    private Slider numHandsSlider; // Reference to the Slider UI element
    
    [SerializeField, Range(1, 2)]
    private int numHandsToTrack = 1; // Number of hands to detect/track

    [SerializeField]
    private bool useLandmarkToDetection = true;

    [Header("UI")]
    [SerializeField]
    private RawImage inputView = null;
    [SerializeField]

    private PalmDetect palmDetect;
    private HandLandmarkDetect landmarkDetect;

    private readonly Vector3[] rtCorners = new Vector3[4];
    private PrimitiveDraw draw;
    private List<PalmDetect.Result> palmResults = new List<PalmDetect.Result>();
    private List<HandLandmarkDetect.Result> landmarkResults = new List<HandLandmarkDetect.Result>();

    private Material previewMaterial;

    private void Start()
    {
        palmDetect = new PalmDetect(palmModelFile);
        landmarkDetect = new HandLandmarkDetect(landmarkModelFile);
        Debug.Log($"landmark dimension: {landmarkDetect.Dim}");

        draw = new PrimitiveDraw();

        previewMaterial = new Material(Shader.Find("Hidden/TFLite/InputMatrixPreview"));
        inputView.material = previewMaterial;

        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.AddListener(OnTextureUpdate);
        }

        numHandsSlider.onValueChanged.AddListener(UpdateNumHandsToTrack);
        UpdateNumHandsToTrack(numHandsSlider.value);
    }

    private void OnDestroy()
    {
        if (TryGetComponent(out VirtualTextureSource source))
        {
            source.OnTexture.RemoveListener(OnTextureUpdate);
        }
        palmDetect?.Dispose();
        landmarkDetect?.Dispose();

        numHandsSlider.onValueChanged.RemoveListener(UpdateNumHandsToTrack);

    }
    private void UpdateNumHandsToTrack(float value)
    {
        numHandsToTrack = (int)value;
    }
    private void Update()
    {

        DrawPalms(palmResults, Color.green);

        if (landmarkResults != null)
        {
            for (int i = 0; i < landmarkResults.Count; i++)
            {
                var result = landmarkResults[i];
                if (result != null && result.score > 0.2f)
                {
                    Color color = (i % 2 == 0) ? Color.blue : Color.red;
                    DrawJoints(result.keypoints, color);
                }
            }
        }
    }

    private void OnTextureUpdate(Texture texture)
    {
        bool needPalmDetect = palmResults == null || palmResults.Count < numHandsToTrack || !useLandmarkToDetection;

        if (needPalmDetect)
        {
            palmDetect.Run(texture);

            inputView.texture = texture;
            previewMaterial.SetMatrix("_TransformMatrix", palmDetect.InputTransformMatrix);
            inputView.rectTransform.GetWorldCorners(rtCorners);

            palmResults = palmDetect.GetResults(0.8f, 0.5f);

            Debug.Log($"Number of palms detected: {palmResults.Count}");
            
  
            if (palmResults.Count <= 0)
            {
                return;
            }
        }

        landmarkResults.Clear();
        foreach (var palmResult in palmResults)
        {
            landmarkDetect.Palm = palmResult;
            landmarkDetect.Run(texture);
            var result = landmarkDetect.GetResult();

            if (result.score >= 0.5f)
            {
                landmarkResults.Add(result.Clone());
            }
        }

        Debug.Log($"Number of landmarks detected: {landmarkResults.Count}");

        if (useLandmarkToDetection)
        {
            palmResults.Clear();
            foreach (var result in landmarkResults)
            {
                LogHandLandmarkCentroid(result);
                palmResults.Add(result.ToDetection());
            }
        }


        
    }

    private void LogHandLandmarkCentroid(HandLandmarkDetect.Result result)
    {
        // Calculate the centroid of the hand landmarks
        Vector3 centroid = Vector3.zero;
        foreach (var joint in result.keypoints)
        {
            centroid += joint;
        }
        centroid /= result.keypoints.Length;

        // Log the centroid to the console
        Debug.Log($"Hand landmark centroid: ({centroid.x}, {centroid.y}, {centroid.z})");
    }
    private void LogPalmCentroids(List<PalmDetect.Result> palms)
    {
        string centroids = "Palm centroids: ";
        foreach (var palm in palms)
        {
            var centroid = palm.rect.center;
            centroids += $"({centroid.x}, {centroid.y}), ";
        }
        Debug.Log(centroids.TrimEnd(' ', ','));
    }
    private void DrawPalms(List<PalmDetect.Result> palms, Color color)
    {
        if (palms == null || palms.Count == 0)
        {
            return;
        }
        float3 min = rtCorners[0];
        float3 max = rtCorners[2];

        draw.color = color;
        foreach (var palm in palms)
        {
            draw.Rect(MathTF.Lerp((Vector3)min, (Vector3)max, palm.rect.FlipY()), 0.02f, min.z);

            foreach (Vector2 kp in palm.keypoints)
            {
                draw.Point(math.lerp(min, max, new float3(kp.x, 1 - kp.y, 0)), 0.05f);
            }
        }
        draw.Apply();
    }

    private void DrawJoints(Vector3[] joints, Color color)
    {
        draw.color = color;

        float3 min = rtCorners[0];
        float3 max = rtCorners[2];

        float zScale = max.x - min.x;
        Vector3[] worldJoints = new Vector3[HandLandmarkDetect.JOINT_COUNT]; // Ensure it's defined within the method
        for (int i = 0; i < HandLandmarkDetect.JOINT_COUNT; i++)
        {
            float3 p0 = joints[i];
            float3 p1 = math.lerp(min, max, p0);
            p1.z += (p0.z - 0.5f) * zScale;
            worldJoints[i] = p1;
        }

        for (int i = 0; i < HandLandmarkDetect.JOINT_COUNT; i++)
        {
            draw.Cube(worldJoints[i], 0.1f);
        }

        var connections = HandLandmarkDetect.CONNECTIONS;
        for (int i = 0; i < connections.Length; i += 2)
        {
            draw.Line3D(
                worldJoints[connections[i]],
                worldJoints[connections[i + 1]],
                0.05f);
        }

        draw.Apply();
    }
}
