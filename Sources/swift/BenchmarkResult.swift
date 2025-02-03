//
//  BenchmarkResult 2.swift
//  llamacpp_swift
//
//  Created by Qi Wang on 2025-02-03.
//
import Foundation

public class BenchmarkResult {
    let promptProcessingTime: Double
    let tokenGenerationTime: Double
    let promptStdDev: Double
    let tokenStdDev: Double
    let modelDescription: String
    let modelSize: String
    let parameterCount: String
    let backend: String
    
    init(
        promptProcessingTime: Double,
        tokenGenerationTime: Double,
        promptStdDev: Double,
        tokenStdDev: Double,
        modelDescription: String,
        modelSize: String,
        parameterCount: String,
        backend: String = "Metal"
    ) {
        self.promptProcessingTime = promptProcessingTime
        self.tokenGenerationTime = tokenGenerationTime
        self.promptStdDev = promptStdDev
        self.tokenStdDev = tokenStdDev
        self.modelDescription = modelDescription
        self.modelSize = modelSize
        self.parameterCount = parameterCount
        self.backend = backend
    }

    func to_string() -> String {
        var result = "| model | size | params | backend | test | t/s |\n"
        result += String(format: "| %@ | %@ | %@ | %@ | %.2f | %.2f |\n",
                        modelDescription,
                        modelSize,
                        parameterCount,
                        backend,
                        promptProcessingTime,
                        tokenGenerationTime)
        return result
    }
}
