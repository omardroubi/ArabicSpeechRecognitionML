//
//  ViewController.swift
//  ArabicSpeechNumbers
//
//  Created by Omar Droubi on 1/27/20.
//  Copyright © 2020 Omar Droubi. All rights reserved.
//

import UIKit
import AVFoundation
import CoreML

class ViewController: UIViewController, AVAudioRecorderDelegate {

    @IBOutlet weak var transcriptionLabel: UILabel!
    @IBOutlet weak var transcriptionLabel2: UILabel!
    
    private let audioEngine = AVAudioEngine()
    let model = ArabicSpeechRecognitionNumbers()
        
    public func getVolume(from buffer: AVAudioPCMBuffer, bufferSize: Int) -> Float {
        guard let channelData = buffer.floatChannelData?[0] else {
            return 0
        }

        let channelDataArray = Array(UnsafeBufferPointer(start:channelData, count: bufferSize))

        var outEnvelope = [Float]()
        var envelopeState:Float = 0
        let envConstantAtk:Float = 0.16
        let envConstantDec:Float = 0.003

        for sample in channelDataArray {
            let rectified = abs(sample)

            if envelopeState < rectified {
                envelopeState += envConstantAtk * (rectified - envelopeState)
            } else {
                envelopeState += envConstantDec * (rectified - envelopeState)
            }
            outEnvelope.append(envelopeState)
        }

        // 0.007 is the low pass filter to prevent
        // getting the noise entering from the microphone
        if let maxVolume = outEnvelope.max(),
            maxVolume > Float(0.015) {
            return maxVolume
        } else {
            return 0.0
        }
    }
    
    override func viewDidLoad() {
        // start recording

        let audioSession = AVAudioSession.sharedInstance()
        
        do {
            try audioSession.setCategory(AVAudioSession.Category.record)
            try audioSession.setMode(AVAudioSession.Mode.measurement)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("audioSession properties weren't set because of an error.")
        }
                
        guard let inputNode: AVAudioNode = self.audioEngine.inputNode else {
            fatalError("Audio engine has no input node")
        }
    
        let recordingFormat = inputNode.outputFormat(forBus: 0)
                
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat, block: {buffer, when in
            
            do {
            // Convert Buffer to MLMultiArray for input to CoreML Model
            
                guard let bufferData = try buffer.floatChannelData else {
               fatalError("Can not get a float handle to buffer")
            }
                
            // Important in order to recognize if person is talking !!!
            print(self.getVolume(from: buffer, bufferSize: 1024))
                

                if (self.getVolume(from: buffer, bufferSize: 1024) > 1) {
            // Chunk data and set to CoreML model
            let windowSize = 15600
            guard let audioData = try? MLMultiArray(shape:[windowSize as NSNumber],
                                                    dataType:MLMultiArrayDataType.float32)
                                                    else {
               fatalError("Can not create MLMultiArray")
            }
            
            var results = [Dictionary<String, Double>]()
            let frameLength = Int(buffer.frameLength)
            var audioDataIndex = 0

            // Iterate over all the samples, chunking calls to analyze every 15600
            for i in 0..<frameLength {
                audioData[audioDataIndex] = NSNumber.init(value: bufferData[0][i])
                if audioDataIndex >= windowSize {
                    let modelInput = ArabicSpeechRecognitionNumbersInput(audioSamples: audioData)

                    guard let modelOutput = try? self.model.prediction(input: modelInput) else {
                        fatalError("Error calling predict")
                    }
                    results.append(modelOutput.classLabelProbs)
                                    var maxLabel = ""
                    var maxOutput: Double = 0.0
                    
                    var maxLabel2 = ""
                    var maxOutput2: Double = 0.0
                    
                    for output in modelOutput.classLabelProbs {
                        if output.value > maxOutput && output.value > 0.8 {
                            maxLabel = output.key
                            maxOutput = output.value
                        }
                    }
                    for output in modelOutput.classLabelProbs {
                        if output.value > maxOutput2 && output.value < maxOutput && output.value > 0.3 {
                            maxLabel2 = output.key
                            maxOutput2 = output.value
                        }
                    }
                                        
                    if (maxOutput > 0 || maxOutput2 > 0) {
                        DispatchQueue.main.async {
                            self.transcriptionLabel.text = maxLabel + " " + String(maxOutput)
                            if (maxOutput2 > 0) {
                                self.transcriptionLabel2.text = maxLabel2 + " " + String(maxOutput2)
                            }
                        }
                    }
                    
                    audioDataIndex = 0
                } else {
                    audioDataIndex += 1
                }
            }
            
            // Handle remainder by passing with zero
            if audioDataIndex > 0 {
                for audioDataIndex in audioDataIndex...windowSize {
                    audioData[audioDataIndex] = 0
                }
                let modelInput = ArabicSpeechRecognitionNumbersInput(audioSamples: audioData)

                guard let modelOutput = try? self.model.prediction(input: modelInput) else {
                    fatalError("Error calling predict")
                }
                results.append(modelOutput.classLabelProbs)
                var maxLabel = ""
                var maxOutput: Double = 0.0
                
                var maxLabel2 = ""
                var maxOutput2: Double = 0.0
                
                for output in modelOutput.classLabelProbs {
                    if output.value > maxOutput && output.value > 0.8 {
                        maxLabel = output.key
                        maxOutput = output.value
                    }
                }
                for output in modelOutput.classLabelProbs {
                    if output.value > maxOutput2 && output.value < maxOutput && output.value > 0.3 {
                        maxLabel2 = output.key
                        maxOutput2 = output.value
                    }
                }
                
                
                if (maxOutput > 0 || maxOutput2 > 0) {
                    DispatchQueue.main.async {
                        self.transcriptionLabel.text = maxLabel + " %" + String(Int(maxOutput * 100))
                        
                        if (maxOutput2 > 0) {
                            self.transcriptionLabel2.text = maxLabel2 + " %" + String(Int(maxOutput2 * 100))
                        }
                    }
                }
            }
                
                }

        
            } catch {
                print("hello")
            }
            
        })
        self.audioEngine.prepare()
        
        do {
            try self.audioEngine.start()
        } catch {
            print("audioEngine couldn't start because of an error.")
        }
        
        }
    
}
