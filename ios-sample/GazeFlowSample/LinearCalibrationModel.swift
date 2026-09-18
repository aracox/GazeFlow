import Foundation

/// Ridge-regularized linear regression fitting screen position from
/// ARKit's lookAtPoint, in place of a0/model.py's sklearn Ridge pipeline
/// (no scikit-learn on iOS). Only 2 input features (lookAtX, lookAtY) here
/// vs the Python model's 17 geometric features, since ARKit's lookAtPoint
/// is already a face-relative gaze estimate rather than raw landmarks --
/// much less feature engineering needed, and a plain 3x3 closed-form solve
/// (bias + 2 weights) is enough; no need for the alpha-selection-via-CV
/// machinery that model matters for on a 17-dimensional feature space.
struct LinearCalibrationModel {
    private let weightsX: SIMD3<Double>  // [bias, w_lookAtX_z, w_lookAtY_z] -> predicted screen x_norm
    private let weightsY: SIMD3<Double>  // -> predicted screen y_norm

    // lookAtX/lookAtY are in meters and tiny (typically +-0.01...0.05), so a
    // ridge penalty on the order of 1.0 would completely swamp the raw
    // signal and shrink the fit toward a near-constant predictor. Standardize
    // to zero mean / unit variance first (same principle as a0/model.py's
    // StandardScaler + Ridge on the Mac) so a ridgeLambda of 1.0 is sane.
    private let meanX: Double
    private let meanY: Double
    private let stdX: Double
    private let stdY: Double

    /// samples: (lookAtX, lookAtY, targetXNorm, targetYNorm)
    init?(samples: [(Float, Float, Float, Float)], ridgeLambda: Double = 1.0) {
        guard samples.count >= 3 else { return nil }

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
        var atbX = [Double](repeating: 0, count: 3)
        var atbY = [Double](repeating: 0, count: 3)

        for (lx, ly, tx, ty) in samples {
            let zx = (Double(lx) - meanX) / stdX
            let zy = (Double(ly) - meanY) / stdY
            let row = [1.0, zx, zy]
            for i in 0..<3 {
                for j in 0..<3 {
                    ata[i][j] += row[i] * row[j]
                }
                atbX[i] += row[i] * Double(tx)
                atbY[i] += row[i] * Double(ty)
            }
        }
        for i in 0..<3 {
            ata[i][i] += ridgeLambda
        }

        guard let wx = Self.solve3x3(ata, atbX), let wy = Self.solve3x3(ata, atbY) else { return nil }
        weightsX = SIMD3(wx[0], wx[1], wx[2])
        weightsY = SIMD3(wy[0], wy[1], wy[2])
    }

    func predict(lookAtX: Float, lookAtY: Float) -> (x: Float, y: Float) {
        let zx = (Double(lookAtX) - meanX) / stdX
        let zy = (Double(lookAtY) - meanY) / stdY
        let row = SIMD3<Double>(1.0, zx, zy)
        let x = (row * weightsX).sum()
        let y = (row * weightsY).sum()
        return (Float(x), Float(y))
    }

    /// Solves the 3x3 linear system a*w = b via Gaussian elimination with
    /// partial pivoting. Returns nil if `a` is singular.
    private static func solve3x3(_ a: [[Double]], _ b: [Double]) -> [Double]? {
        var m = a
        var rhs = b
        let n = 3

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
