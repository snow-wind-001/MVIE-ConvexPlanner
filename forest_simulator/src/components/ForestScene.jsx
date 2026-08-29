import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { samplePolyline } from '../lib/path.js'

const WORLD_LENGTH = 36

function worldPoint(point) {
  return new THREE.Vector3(point[0], point[2], point[1] - WORLD_LENGTH / 2)
}

function addPolyline(group, path, material, radius = 0.025) {
  if (!path || path.length < 2) return
  for (let index = 0; index < path.length - 1; index += 1) {
    const start = worldPoint(path[index])
    const end = worldPoint(path[index + 1])
    const midpoint = start.clone().add(end).multiplyScalar(0.5)
    const direction = end.clone().sub(start)
    const geometry = new THREE.CylinderGeometry(radius, radius, direction.length(), 8)
    const segment = new THREE.Mesh(geometry, material)
    segment.position.copy(midpoint)
    segment.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      direction.clone().normalize(),
    )
    group.add(segment)
  }
}

function addCapsule(group, branch, material, radiusOffset = 0, castShadow = true) {
  const start = worldPoint(branch.start)
  const end = worldPoint(branch.end)
  const direction = end.clone().sub(start)
  const radius = branch.radius + radiusOffset
  if (direction.lengthSq() < 1e-12) return
  const midpoint = start.clone().add(end).multiplyScalar(0.5)
  const cylinder = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, direction.length(), 10),
    material,
  )
  cylinder.position.copy(midpoint)
  cylinder.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.clone().normalize(),
  )
  cylinder.castShadow = castShadow
  group.add(cylinder)
  for (const endpoint of [start, end]) {
    const cap = new THREE.Mesh(new THREE.SphereGeometry(radius, 10, 7), material)
    cap.position.copy(endpoint)
    cap.castShadow = false
    group.add(cap)
  }
}

function createDrone() {
  const drone = new THREE.Group()
  const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0xe8f7ee, metalness: 0.55, roughness: 0.3 })
  const accentMaterial = new THREE.MeshStandardMaterial({ color: 0x66e49a, emissive: 0x173c27 })
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.14, 0.28), bodyMaterial)
  drone.add(body)
  const armGeometry = new THREE.BoxGeometry(0.86, 0.035, 0.045)
  for (const rotation of [Math.PI / 4, -Math.PI / 4]) {
    const arm = new THREE.Mesh(armGeometry, bodyMaterial)
    arm.rotation.y = rotation
    drone.add(arm)
  }
  const rotorGeometry = new THREE.CylinderGeometry(0.14, 0.14, 0.018, 24)
  for (const [x, z] of [[0.31, 0.31], [-0.31, 0.31], [0.31, -0.31], [-0.31, -0.31]]) {
    const rotor = new THREE.Mesh(rotorGeometry, accentMaterial)
    rotor.position.set(x, 0.04, z)
    drone.add(rotor)
  }
  drone.scale.setScalar(0.8)
  return drone
}

function createMarker(color) {
  const group = new THREE.Group()
  const material = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.24 })
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.34, 0.045, 10, 32), material)
  ring.rotation.x = Math.PI / 2
  group.add(ring)
  const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 1.2, 8), material)
  beacon.position.y = 0.6
  group.add(beacon)
  return group
}

export default function ForestScene({ scenario, progress, showReference, showSafety, cameraMode }) {
  const mountRef = useRef(null)
  const runtimeRef = useRef(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return undefined
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x07110e)
    scene.fog = new THREE.FogExp2(0x07110e, 0.021)

    const camera = new THREE.PerspectiveCamera(47, mount.clientWidth / mount.clientHeight, 0.1, 120)
    camera.position.set(13.5, 12.0, 19.5)
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75))
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFShadowMap
    renderer.outputColorSpace = THREE.SRGBColorSpace
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.target.set(0, 2.8, 0)
    controls.minDistance = 5
    controls.maxDistance = 55
    controls.maxPolarAngle = Math.PI / 2.02

    scene.add(new THREE.HemisphereLight(0xb7d7ca, 0x07110e, 2.2))
    const sun = new THREE.DirectionalLight(0xf0f6df, 3.1)
    sun.position.set(-8, 18, -10)
    sun.castShadow = true
    sun.shadow.mapSize.set(2048, 2048)
    sun.shadow.camera.left = -18
    sun.shadow.camera.right = 18
    sun.shadow.camera.top = 25
    sun.shadow.camera.bottom = -25
    scene.add(sun)

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(18, 42),
      new THREE.MeshStandardMaterial({ color: 0x102219, roughness: 1 }),
    )
    ground.rotation.x = -Math.PI / 2
    ground.receiveShadow = true
    scene.add(ground)
    const grid = new THREE.GridHelper(42, 42, 0x2f5d44, 0x183426)
    grid.position.y = 0.012
    scene.add(grid)

    const treeGroup = new THREE.Group()
    const trunkMaterial = new THREE.MeshStandardMaterial({ color: 0x74543a, roughness: 0.92 })
    const branchMaterial = new THREE.MeshStandardMaterial({ color: 0x68482f, roughness: 0.94 })
    const canopyMaterials = [0x24583a, 0x2e6846, 0x347650].map(
      (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.9 }),
    )
    const dynamicTrunk = new THREE.MeshStandardMaterial({ color: 0xf0a24a, emissive: 0x5b2a0b, emissiveIntensity: 0.5 })
    const dynamicCanopy = new THREE.MeshStandardMaterial({ color: 0xc86e32, emissive: 0x5b2411, emissiveIntensity: 0.3 })
    scenario.trees.forEach((tree, index) => {
      const center = worldPoint(tree.center)
      const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(tree.radius, tree.radius, tree.height, 12),
        tree.dynamic ? dynamicTrunk : trunkMaterial,
      )
      trunk.position.copy(center)
      trunk.castShadow = true
      trunk.receiveShadow = true
      treeGroup.add(trunk)
      tree.branches.forEach((branch) => {
        addCapsule(
          treeGroup,
          branch,
          tree.dynamic ? dynamicTrunk : branchMaterial,
        )
      })
      tree.canopy_spheres.forEach((crown) => {
        const canopy = new THREE.Mesh(
          new THREE.SphereGeometry(crown.radius, 16, 11),
          tree.dynamic ? dynamicCanopy : canopyMaterials[index % canopyMaterials.length],
        )
        canopy.position.copy(worldPoint(crown.center))
        canopy.castShadow = true
        treeGroup.add(canopy)
      })
    })
    scene.add(treeGroup)

    if (showSafety) {
      const dynamic = scenario.trees.find((tree) => tree.dynamic)
      if (dynamic) {
        const safetyMaterial = new THREE.MeshBasicMaterial({
          color: 0xffbe63,
          transparent: true,
          opacity: 0.12,
          side: THREE.DoubleSide,
          depthWrite: false,
        })
        const safety = new THREE.Mesh(
          new THREE.CylinderGeometry(
            dynamic.radius + 0.3,
            dynamic.radius + 0.3,
            dynamic.height + 0.6,
            32,
          ),
          safetyMaterial,
        )
        const safetyCenter = worldPoint(dynamic.center)
        safetyCenter.y += 0.3
        safety.position.copy(safetyCenter)
        scene.add(safety)
        dynamic.branches.forEach((branch) => {
          addCapsule(scene, branch, safetyMaterial, 0.3, false)
        })
        dynamic.canopy_spheres.forEach((crown) => {
          const crownSafety = new THREE.Mesh(
            new THREE.SphereGeometry(crown.radius + 0.3, 16, 11),
            safetyMaterial,
          )
          crownSafety.position.copy(worldPoint(crown.center))
          scene.add(crownSafety)
        })
      }
    }

    const pathGroup = new THREE.Group()
    if (showReference) {
      const normalReference = new THREE.MeshBasicMaterial({ color: 0x4f91a6, transparent: true, opacity: 0.72 })
      const collisionReference = new THREE.MeshBasicMaterial({ color: 0xf26f5b })
      for (let index = 0; index < scenario.reference_path.length - 1; index += 1) {
        addPolyline(
          pathGroup,
          [scenario.reference_path[index], scenario.reference_path[index + 1]],
          scenario.reference_collision_segments.includes(index) ? collisionReference : normalReference,
          scenario.reference_collision_segments.includes(index) ? 0.055 : 0.022,
        )
      }
    }
    if (scenario.realtime_path) {
      addPolyline(
        pathGroup,
        scenario.realtime_path,
        new THREE.MeshBasicMaterial({ color: 0x69e79d }),
        0.055,
      )
      const waypointMaterial = new THREE.MeshBasicMaterial({ color: 0xb7f6cd })
      scenario.realtime_path.slice(1, -1).forEach((point) => {
        const marker = new THREE.Mesh(new THREE.SphereGeometry(0.085, 12, 8), waypointMaterial)
        marker.position.copy(worldPoint(point))
        pathGroup.add(marker)
      })
    }
    scene.add(pathGroup)

    const startMarker = createMarker(0x65e39a)
    startMarker.position.copy(worldPoint(scenario.start))
    scene.add(startMarker)
    const goalMarker = createMarker(0x6eb6ff)
    goalMarker.position.copy(worldPoint(scenario.goal))
    scene.add(goalMarker)

    const drone = createDrone()
    drone.position.copy(worldPoint(scenario.start))
    scene.add(drone)

    let previousTime = performance.now()
    let animationFrame
    const animate = (timestamp = performance.now()) => {
      animationFrame = requestAnimationFrame(animate)
      const delta = Math.min((timestamp - previousTime) / 1000, 0.05)
      previousTime = timestamp
      drone.children.slice(3).forEach((rotor) => {
        rotor.rotation.y += delta * 18
      })
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    const resizeObserver = new ResizeObserver(() => {
      if (!mount.clientWidth || !mount.clientHeight) return
      camera.aspect = mount.clientWidth / mount.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(mount.clientWidth, mount.clientHeight)
    })
    resizeObserver.observe(mount)
    runtimeRef.current = { camera, controls, drone }

    return () => {
      resizeObserver.disconnect()
      cancelAnimationFrame(animationFrame)
      controls.dispose()
      renderer.dispose()
      scene.traverse((object) => {
        object.geometry?.dispose?.()
        if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose())
        else object.material?.dispose?.()
      })
      mount.removeChild(renderer.domElement)
      runtimeRef.current = null
    }
  }, [scenario, showReference, showSafety])

  useEffect(() => {
    const runtime = runtimeRef.current
    if (!runtime) return
    const activePath = scenario.realtime_path
    const point = samplePolyline(activePath, progress)
    if (!point) return
    const position = worldPoint(point)
    runtime.drone.position.copy(position)
    if (cameraMode === 'follow') {
      const desired = position.clone().add(new THREE.Vector3(5.0, 3.0, 6.0))
      runtime.camera.position.lerp(desired, 0.08)
      runtime.controls.target.lerp(position, 0.12)
    }
  }, [cameraMode, progress, scenario])

  useEffect(() => {
    const runtime = runtimeRef.current
    if (!runtime || cameraMode !== 'overview') return
    runtime.camera.position.set(13.5, 12.0, 19.5)
    runtime.controls.target.set(0, 2.8, 0)
    runtime.controls.update()
  }, [cameraMode])

  return <div className="scene-mount" ref={mountRef} aria-label="无人机穿越树林三维仿真场景" />
}
