export function pathLength(path) {
  if (!path || path.length < 2) return 0
  let total = 0
  for (let index = 0; index < path.length - 1; index += 1) {
    const a = path[index]
    const b = path[index + 1]
    total += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2])
  }
  return total
}

export function samplePolyline(path, progress) {
  if (!path?.length) return null
  if (path.length === 1) return path[0]
  const lengths = []
  let total = 0
  for (let index = 0; index < path.length - 1; index += 1) {
    const a = path[index]
    const b = path[index + 1]
    const length = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2])
    lengths.push(length)
    total += length
  }
  let target = Math.max(0, Math.min(1, progress)) * total
  for (let index = 0; index < lengths.length; index += 1) {
    if (target <= lengths[index] || index === lengths.length - 1) {
      const ratio = lengths[index] === 0 ? 0 : target / lengths[index]
      return path[index].map(
        (value, axis) => value + (path[index + 1][axis] - value) * ratio,
      )
    }
    target -= lengths[index]
  }
  return path[path.length - 1]
}
