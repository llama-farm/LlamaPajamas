'use client'

import { useState } from 'react'
import ModelsPanel from '@/components/ModelsPanel'
import QuantizePanel from '@/components/QuantizePanel'
import ExportPanel from '@/components/ExportPanel'
import EvaluatePanel from '@/components/EvaluatePanel'
import BatchPanel from '@/components/BatchPanel'
import ServerPanel from '@/components/ServerPanel'
import InferencePanel from '@/components/InferencePanel'
import SettingsPanel from '@/components/SettingsPanel'

type Tab = 'models' | 'quantize' | 'export' | 'evaluate' | 'batch' | 'server' | 'inference' | 'settings'

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>('settings')

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: 'settings', label: 'Settings', icon: '⚙️' },
    { id: 'models', label: 'Models', icon: '📁' },
    { id: 'quantize', label: 'Quantize', icon: '⚡' },
    { id: 'export', label: 'Export', icon: '📤' },
    { id: 'evaluate', label: 'Evaluate', icon: '📊' },
    { id: 'batch', label: 'Batch', icon: '🔄' },
    { id: 'server', label: 'Server', icon: '🚀' },
    { id: 'inference', label: 'Inference', icon: '💬' },
  ]

  return (
    <main className="min-h-screen p-8 bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold mb-2">LlamaPajamas - Simple UI</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Quantize, export, evaluate, and run models - Complete workflow interface
          </p>
        </header>

        {/* Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
          <nav className="flex gap-2 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 font-medium border-b-2 transition-colors whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                <span className="mr-1">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          {activeTab === 'settings' && <SettingsPanel />}
          {activeTab === 'models' && <ModelsPanel />}
          {activeTab === 'quantize' && <QuantizePanel />}
          {activeTab === 'export' && <ExportPanel />}
          {activeTab === 'evaluate' && <EvaluatePanel />}
          {activeTab === 'batch' && <BatchPanel />}
          {activeTab === 'server' && <ServerPanel />}
          {activeTab === 'inference' && <InferencePanel />}
        </div>
      </div>
    </main>
  )
}
