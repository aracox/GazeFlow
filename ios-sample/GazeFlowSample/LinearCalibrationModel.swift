import Foundation

private extension Double {
    func clamped(to range: ClosedRange<Double>) -> Double {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

/// Fits screen position from ARKit's lookAtPoint.
///
/// X uses piecewise-linear interpolation directly between the calibrated
/// points, not a single regression curve. A single polynomial fit (first a
/// line, then a line + quadratic term) forces one global tradeoff across
/// the whole screen: enough curvature to correct the compression seen near
/// one edge (e.g. digits 4-5 selecting unreliably) either wasn't enough to
/// help there, or -- once strong enough to help -- overshot past the
/// *other* edge (e.g. the gaze cursor flying off-screen while looking at
/// digit 0), and reining that back in with heavier regularization undid
/// the original fix. Piecewise-linear interpolation instead guarantees an
/// exact match at every calibrated target (each segment is anchored by
/// its own two neighboring points, so correcting one edge can't distort
/// another), only extrapolating gently (constant slope, not accelerating)
/// beyond the collected range, with a final clamp to keep results on
/// screen.
///
/// Y is never used for actual selection in this app -- every mode's real
/// targets sit on one horizontal row, so box/zone selection only reads
/// predicted X -- so it keeps a plain standardized ridge-regularized
/// linear fit (same principle as a0/model.py's sklearn Ridge pipeline,
/// hand-rolled since there's no scikit-learn on iOS); it only drives the
/// optional debug gaze dot's vertical position.
struct LinearCalibrationModel {
    private struct Anchor { let rawX: Double; let targetX: Double }

    private let xAnchors: [Anchor]
    private let weightsY: SIMD3<Double>  // [bias, w_lookAtX_z, w_lookAtY_z] -> predicted screen y_norm

    // lookAtX/lookAtY are in meters and tiny (typically +-0.01...0.05), so a
    // ridge penalty on the order of 1.0 would completely swamp the raw
    // signal and shrink the Y fit toward a near-constant predictor.
    // Standardize to zero mean / unit variance first (same principle as
    // a0/model.py's StandardScaler + Ridge on the Mac) so a ridgeLambda of
    // 1.0 is sane.
    private let meanX: Double
    private let meanY: Double
    private let stdX: Double
    private let stdY: Double

    /// samples: (lookAtX, lookAtY, targetXNorm, targetYNorm)
    init?(samples: [(Float, Float, Float, Float)], ridgeLambda: Double = 1.0) {
        guard samples.count >= 3 else { return nil }

        // Average raw lookAtX per distinct calibrated target X (points that
        // share the same target column, like Numbers' off-row top/bottom
        // stability points sharing x=0.5 with no digit, get merged into one
        // anchor rather than needing an exact digit at every column).
        var sumsByTargetX: [Float: (sum: Double, count: Int)] = [:]
        for (lx, _, tx, _) in samples {
            var entry = sumsByTargetX[tx] ?? (0, 0)
            entry.sum += Double(lx)
            entry.count += 1
            sumsByTargetX[tx] = entry
        }
        guard sumsByTargetX.count >= 2 else { return nil }
        xAnchors = sumsByTargetX
            .map { tx, entry in Anchor(rawX: entry.sum / Double(entry.count), targetX: Double(tx)) }
            .sorted { $0.rawX < $1.rawX }

        let n = Double(samples.count)
        let sumX = samples.reduce(0.0) { $0 + Double($1.0) }
        let sumY = samples.reduce(0.0) { $0 + Double($1.1) }
        let mX = sumX / n
        let mY = sumY / n
        let varX = samples.reduce(0.0) { $0 + (Double($1.0) - mX) * (Double($1.0) - mX) } / n
        let varY = samples.reduce(0.0) { $0 + (Double($1.1) - mY) * (Double($1.1) - mY) } / n
        let sX = max(sqrt(varX), 1e-6)
        let sY = max(sqrt(varY), 1e-6)
        meanX = mX; meanY = mY; stdX = sX; stdY = sY

        var ata = [[Double]](repeating: [Double](repeating: 0, count: 3), count: 3)
        var atbY = [Double](repeating: 0, count: 3)
        for (lx, ly, _, ty) in samples {
            let zx = (Double(lx) - meanX) / stdX
            let zy = (Double(ly) - meanY) / stdY
            let row = [1.0, zx, zy]
            for i in 0..<3 {
                for j in 0..<3 { ata[i][j] += row[i] * row[j] }
                atbY[i] += row[i] * Double(ty)
            }
        }
        for i in 0..<3 { ata[i][i] += ridgeLambda }

        guard let wy = Self.solve(ata, atbY) else { return nil }
        weightsY = SIMD3(wy[0], wy[1], wy[2])
    }

    func predict(lookAtX: Float, lookAtY: Float) -> (x: Float, y: Float) {
        let x = Self.interpolateX(Double(lookAtX), anchors: xAnchors)

        let zx = (Double(lookAtX) - meanX) / stdX
        let zy = (Double(lookAtY) - meanY) / stdY
        let y = (SIMD3(1.0, zx, zy) * weightsY).sum()

        // Screen position is only ever meaningfully within [0,1] -- clamp
        // so a beyond-calibration-range reading pins the cursor to the
        // edge instead of drawing (and selecting against) a point off the
        // visible screen.
        return (Float(x.clamped(to: 0...1)), Float(y.clamped(to: 0...1)))
    }

    private static func interpolateX(_ rawX: Double, anchors: [Anchor]) -> Double {
        guard let first = anchors.first, let last = anchors.last, anchors.count >= 2 else {
            return anchors.first?.targetX ?? 0.5
        }

        func segmentTarget(_ a: Anchor, _ b: Anchor, at x: Double) -> Double {
            let span = b.rawX - a.rawX
            guard abs(span) > 1e-9 else { return a.targetX }
            let t = (x - a.rawX) / span
            return a.targetX + t * (b.targetX - a.targetX)
        }

        if rawX <= first.rawX { return segmentTarget(anchors[0], anchors[1], at: rawX) }
        if rawX >= last.rawX { return segmentTarget(anchors[anchors.count - 2], anchors[anchors.count - 1], at: rawX) }

        for i in 0..<(anchors.count - 1) where rawX <= anchors[i + 1].rawX {
            return segmentTarget(anchors[i], anchors[i + 1], at: rawX)
        }
        return last.targetX
    }

    /// Solves the NxN linear system a*w = b via Gaussian elimination with
    /// partial pivoting. Returns nil if `a` is singular.
    private static func solve(_ a: [[Double]], _ b: [Double]) -> [Double]? {
        var m = a
        var rhs = b
        let n = b.count

        for col in 0..<n {
            var pivotRow = col
            for row in (col + 1)..<n where abs(m[row][col]) > abs(m[pivotRow][col]) {
                pivotRow = row
            }
            if abs(m[pivotRow][col]) < 1e-9 { return nil }
            if pivotRow != col {
                m.swapAt(col, pivotRow)
                rhs.swapAt(col, pivotRow)
            }
            for row in (col + 1)..<n {
                let factor = m[row][col] / m[col][col]
                for k in col..<n { m[row][k] -= factor * m[col][k] }
                rhs[row] -= factor * rhs[col]
            }
        }

        var w = [Double](repeating: 0, count: n)
        for row in stride(from: n - 1, through: 0, by: -1) {
            var sum = rhs[row]
            for k in (row + 1)..<n { sum -= m[row][k] * w[k] }
            w[row] = sum / m[row][row]
        }
        return w
    }
}
