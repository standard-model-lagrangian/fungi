export type TargetObjectType = 'fungal_hyphae' | 'mixed_fungi_bacteria' | 'bacteria'
export type SegmentationPresetId = TargetObjectType
export type SkeletonizationMode = 'hyphae_only' | 'all_objects'

export interface ObjectClassificationValues {
  hyphae_min_length_px: number
  hyphae_min_aspect_ratio: number
  bacteria_max_length_px: number
  bacteria_max_area_px2: number
  max_component_count_threshold: number
}

export interface BacterialTrackingValues {
  enable_bacterial_tracking: boolean
  max_bacteria_displacement_px: number
  max_track_gap_frames: number
  min_track_length_frames: number
  trajectory_tail_frames: number
  generate_trajectory_overlay_video: boolean
  generate_heatmaps: boolean
  max_objects_per_frame_for_tracking: number
}

export interface SegmentationPresetValues {
  min_object_size_px: number
  hole_fill_area: number
  dilation_radius: number
  min_branch_length_px: number
  skeletonization_mode: SkeletonizationMode
  skeleton_min_object_area_px: number
  use_temporal_continuity: boolean
  repair_disconnected_tubes: boolean
  classification: ObjectClassificationValues
  tracking: BacterialTrackingValues
}

export interface WorkflowFlags {
  enable_skeletonization: boolean
  enable_branch_detection: boolean
  enable_tip_detection: boolean
  enable_tube_repair: boolean
  enable_traversed_persistence: boolean
  enable_hyphal_growth_metrics: boolean
  enable_bacterial_metrics: boolean
  enable_fungal_processing: boolean
  enable_bacterial_processing: boolean
}

export interface SegmentationPreset {
  id: TargetObjectType
  label: string
  values: SegmentationPresetValues
  workflow: WorkflowFlags
}

const FUNGAL_TRACKING: BacterialTrackingValues = {
  enable_bacterial_tracking: false,
  max_bacteria_displacement_px: 20,
  max_track_gap_frames: 2,
  min_track_length_frames: 2,
  trajectory_tail_frames: 20,
  generate_trajectory_overlay_video: true,
  generate_heatmaps: true,
  max_objects_per_frame_for_tracking: 5000,
}

const BACTERIAL_TRACKING: BacterialTrackingValues = {
  enable_bacterial_tracking: true,
  max_bacteria_displacement_px: 20,
  max_track_gap_frames: 2,
  min_track_length_frames: 2,
  trajectory_tail_frames: 20,
  generate_trajectory_overlay_video: true,
  generate_heatmaps: true,
  max_objects_per_frame_for_tracking: 5000,
}

export const SEGMENTATION_PRESETS: SegmentationPreset[] = [
  {
    id: 'fungal_hyphae',
    label: 'Fungal Hyphae',
    values: {
      min_object_size_px: 40,
      hole_fill_area: 200,
      dilation_radius: 8,
      min_branch_length_px: 10,
      skeletonization_mode: 'hyphae_only',
      skeleton_min_object_area_px: 40,
      use_temporal_continuity: true,
      repair_disconnected_tubes: true,
      classification: {
        hyphae_min_length_px: 12,
        hyphae_min_aspect_ratio: 2.5,
        bacteria_max_length_px: 15,
        bacteria_max_area_px2: 200,
        max_component_count_threshold: 5000,
      },
      tracking: FUNGAL_TRACKING,
    },
    workflow: {
      enable_skeletonization: true,
      enable_branch_detection: true,
      enable_tip_detection: true,
      enable_tube_repair: true,
      enable_traversed_persistence: true,
      enable_hyphal_growth_metrics: true,
      enable_bacterial_metrics: false,
      enable_fungal_processing: true,
      enable_bacterial_processing: false,
    },
  },
  {
    id: 'bacteria',
    label: 'Bacteria',
    values: {
      min_object_size_px: 1,
      hole_fill_area: 0,
      dilation_radius: 1,
      min_branch_length_px: 1,
      skeletonization_mode: 'hyphae_only',
      skeleton_min_object_area_px: 1,
      use_temporal_continuity: false,
      repair_disconnected_tubes: false,
      classification: {
        hyphae_min_length_px: 9999,
        hyphae_min_aspect_ratio: 99,
        bacteria_max_length_px: 9999,
        bacteria_max_area_px2: 1_000_000,
        max_component_count_threshold: 8000,
      },
      tracking: BACTERIAL_TRACKING,
    },
    workflow: {
      enable_skeletonization: false,
      enable_branch_detection: false,
      enable_tip_detection: false,
      enable_tube_repair: false,
      enable_traversed_persistence: false,
      enable_hyphal_growth_metrics: false,
      enable_bacterial_metrics: true,
      enable_fungal_processing: false,
      enable_bacterial_processing: true,
    },
  },
  {
    id: 'mixed_fungi_bacteria',
    label: 'Mixed Fungi + Bacteria',
    values: {
      min_object_size_px: 1,
      hole_fill_area: 5,
      dilation_radius: 2,
      min_branch_length_px: 3,
      skeletonization_mode: 'hyphae_only',
      skeleton_min_object_area_px: 15,
      use_temporal_continuity: true,
      repair_disconnected_tubes: false,
      classification: {
        hyphae_min_length_px: 8,
        hyphae_min_aspect_ratio: 2.0,
        bacteria_max_length_px: 12,
        bacteria_max_area_px2: 120,
        max_component_count_threshold: 5000,
      },
      tracking: BACTERIAL_TRACKING,
    },
    workflow: {
      enable_skeletonization: true,
      enable_branch_detection: true,
      enable_tip_detection: true,
      enable_tube_repair: false,
      enable_traversed_persistence: true,
      enable_hyphal_growth_metrics: true,
      enable_bacterial_metrics: true,
      enable_fungal_processing: true,
      enable_bacterial_processing: true,
    },
  },
]

export const SMALL_OBJECT_WARNING_THRESHOLD = 5
export const SMALL_OBJECT_WARNING =
  'Very small object sizes may increase noise and processing time.'

export function presetById(id: string): SegmentationPreset {
  return SEGMENTATION_PRESETS.find((p) => p.id === id) ?? SEGMENTATION_PRESETS[0]
}

export function shouldShowSmallObjectWarning(minObjectSizePx: number): boolean {
  const value = Number(minObjectSizePx)
  return Number.isFinite(value) && value < SMALL_OBJECT_WARNING_THRESHOLD
}

export function supportsBacterialTracking(targetObjectType: TargetObjectType): boolean {
  return targetObjectType === 'bacteria' || targetObjectType === 'mixed_fungi_bacteria'
}
